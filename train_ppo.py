import argparse
import csv
import copy
import json
import math
import multiprocessing as mp
import os
import random
import signal
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from game import (GameEnv, HumanPlayer, Move, RandomPlayer, Card, RANKS, SUITS,
                  POINT_VALUES, SMALL_JOKER, BIG_JOKER, HISTORY_MAXLEN)
from encoder import encode_move, encode_state, move_feature_dim, _min_turns_to_empty
from shaping import scaled_potential, apply_shaping
from endgame import endgame_value_from_env, best_action as endgame_best_action, _key as _eg_key, EndgameSolverPlayer
from mcts import ISMCTSPlayer

# ======================
# Hyperparameters
# ======================

GAMMA = 0.997
LAMBDA = 0.97
LR = 3e-5
EPS_CLIP = 0.12
EPOCHS = 2
MINIBATCH_SIZE = 128
ROLLOUTS_PER_BATCH = 20

# Reverted to the V3 recipe (the policy you liked). The aggressive plateau-pushers
# (higher entropy, adaptive LR to 3e-4, value/KL clipping) were keeping mid-training
# checkpoints under-converged, so they're neutralized here.
VALUE_CLIP = 1e9        # effectively unclipped value loss (V3 behavior)
TARGET_KL = 1e9         # KL early-stop disabled
KL_STOP_FACTOR = 1.5
ADAPTIVE_LR = False     # fixed LR (V3 behavior)
LR_MIN = 1e-5
LR_MAX = 3e-4

ENTROPY_START = 0.010
ENTROPY_END = 0.0015    # V3 floor
VALUE_COEF = 0.75
AUX_COEF = 0.40
LOOKAHEAD_COEF = 0.25

POINT_SCALE = 10.0

# Potential-based reward shaping for control-card economy. beta=0 disables it
# (recovers the original pure-margin reward). See shaping.py for the rationale.
# It is potential-based (policy-invariant) AND annealed to 0 over training, so the
# converged policy optimizes the pure points/win objective with no residual bias.
SHAPING_BETA = 1.0
SHAPING_ANNEAL_EPISODES = 0      # constant shaping (V3 behavior; 0 = no anneal)

# Once the deck is empty the game is perfect information -> solve it EXACTLY (endgame.py)
# and use that value to bootstrap training rollouts. Teaches the policy the true value
# of conserving for the endgame, and the deployed bot plays the run-out optimally.
USE_ENDGAME_SOLVER = True

# ---- Dedicated exploiters ----
# An exploiter is a separate agent trained *only* to beat the current main policy.
# It auto-discovers weaknesses (the things humans find by playtesting) and forces the
# main to patch them. Based on the PSRO / AlphaStar exploiter concept.
EXPLOITER_ENABLED = True
EXPLOITER_TRAIN_FREQ = 50        # train the exploiter every N main episodes
EXPLOITER_TRAIN_STEPS = 30       # rollouts per exploiter update (small: exploit fast)
EXPLOITER_EVAL_GAMES = 60        # games to evaluate exploiter vs. main
EXPLOITER_ADD_THRESHOLD = 0.60   # add to league when it beats main > this WR
EXPLOITER_LR = 3e-4              # higher LR: fast convergence to exploitative policy
EXPLOITER_WIN_SCALE = 20.0       # reward is ±WIN_SCALE for win/loss only (not margin)
EXPLOITER_MAX_IN_LEAGUE = 4      # evict oldest when league is full of exploiters

# Sparse terminal reward (in point-equivalents) for winning vs losing the game.
# Aligns the policy with win-rate (not just margin) -- decisive in the close
# games that occur against strong opponents. 0 disables.
WIN_BONUS = 15.0

HISTORY_MAX_EVENTS = 48
HISTORY_EVENT_DIM = 42

LEAGUE_SIZE = 16        # V3 value
RECENT_POOL_SIZE = 8
FRONTIER_SIZE = 4

# Prioritized fictitious self-play (PFSP): sample opponents we are NOT beating.
PFSP_ETA = 2.0          # V3 value
PFSP_FLOOR = 0.04       # keep a little probability on already-beaten opponents
PFSP_MIN_GAMES = 6      # below this many games an opponent is sampled optimistically
PFSP_DECAY = 0.99       # exponential forgetting so stats track the *current* policy
SELF_SHARE = 0.15       # fraction of games reserved for vanilla self-play
RANDOM_MIN_WEIGHT = 0.02
SNAPSHOT_EVAL_FREQ = 50
EVAL_PRINT_FREQ = 50
# Bigger eval samples: at 18-24 games the gating/best-selection signal was pure
# noise (SE~0.11 at p=0.5), so genuine improvements couldn't register and the
# best-checkpoint score stalled. These ~halve the eval noise.
RANDOM_EVAL_GAMES = 60
BASELINE_EVAL_GAMES = 150
POOL_EVAL_GAMES = 40
RECENT_BEST_GAMES = 120
FRONTIER_EVAL_GAMES = 40
SCRIPTED_EVAL_GAMES = 48

REPLAY_BUFFER_SIZE = 768
REPLAY_MIN_BUFFER = 32
REPLAY_START_EP = 80
REPLAY_MAX_PROB = 0.35

USE_LOOKAHEAD = False    # off by default: expensive on CPU; enable via --lookahead
LOOKAHEAD_START_EP = 40
LOOKAHEAD_MAX_ACTIONS = 8
LOOKAHEAD_ROLLOUTS = 3   # now meaningful: each uses a fresh PIMC determinization
LOOKAHEAD_HORIZON = 10
LOOKAHEAD_TEMP = 0.60

CHECKPOINT_LATEST = "policy_latest.pt"
CHECKPOINT_BEST = "policy_best.pt"
DEFAULT_BASELINE_CHECKPOINT = "policy_latest_bl.pt"

PERF_CSV_PATH = "training_performance.csv"
PERF_PNG_PATH = "training_performance.png"


# ======================
# Device helpers
# ======================


def select_device(requested: str = "auto"):
    requested = (requested or "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested, but no CUDA device is available.")
        device = torch.device("cuda")
    elif requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise ValueError("MPS was requested, but no MPS device is available.")
        device = torch.device("mps")
    elif requested == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"Unknown device request: {requested}")

    if device.type == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    return device


DEVICE = select_device("auto")


# ======================
# History encoding
# ======================

EVENT_KIND_TO_IDX = {"deal": 0, "play": 1, "pass": 2, "hand_end": 3, "draw": 4, "terminal": 5}
MOVE_TYPE_TO_IDX = {"none": 0, "single": 1, "pair": 2, "triple": 3, "straight": 4, "bomb": 5}
RANK_TO_IDX = {r: i for i, r in enumerate(RANKS)}


def encode_history_events(infoset: Dict, max_events: int = HISTORY_MAX_EVENTS) -> np.ndarray:
    p = infoset.get("player_index", 0)
    history = infoset.get("public_history", [])[-max_events:]
    out = np.zeros((len(history), HISTORY_EVENT_DIM), dtype=np.float32)
    for i, event in enumerate(history):
        vec = out[i]
        kind = event.get("kind", "deal")
        vec[EVENT_KIND_TO_IDX.get(kind, 0)] = 1.0
        base = len(EVENT_KIND_TO_IDX)

        actor = event.get("actor", -1)
        if actor == p:
            vec[base + 0] = 1.0
        elif actor == 1 - p:
            vec[base + 1] = 1.0
        else:
            vec[base + 2] = 1.0
        base += 3

        move_type = event.get("move_type", "none")
        vec[base + MOVE_TYPE_TO_IDX.get(move_type, 0)] = 1.0
        base += len(MOVE_TYPE_TO_IDX)

        move_rank = event.get("move_rank", 0)
        if move_rank in RANK_TO_IDX:
            vec[base + RANK_TO_IDX[move_rank]] = 1.0
        base += len(RANKS)

        move_len = event.get("move_len", 0)
        vec[base + 0] = move_len / 5.0
        vec[base + 1] = event.get("points_gained", 0) / 40.0
        vec[base + 2] = event.get("pot_after", event.get("pot", 0)) / 40.0
        vec[base + 3] = event.get("deck_size", infoset.get("deck_size", 0)) / 54.0
        hand_sizes = event.get("hand_sizes", [0, 0])
        vec[base + 4] = hand_sizes[p] / 5.0
        vec[base + 5] = hand_sizes[1 - p] / 5.0
        draw_counts = event.get("draw_counts", [0, 0])
        vec[base + 6] = draw_counts[p] / 5.0
        vec[base + 7] = draw_counts[1 - p] / 5.0
        winner = event.get("winner", -1)
        vec[base + 8] = 1.0 if winner == p else 0.0
        vec[base + 9] = 1.0 if winner == 1 - p else 0.0
        vec[base + 10] = event.get("pass_count", 0) / 2.0
        vec[base + 11] = float(event.get("is_bomb", 0))
    return out


# ======================
# Scripted exploiters
# ======================


def move_points(move):
    if move is None:
        return 0
    cards = move.cards if isinstance(move, Move) else move
    return sum(POINT_VALUES.get(c.rank, 0) for c in cards)


def move_obj(move):
    return move if isinstance(move, Move) else (None if move is None else Move(move))


class TrapBombPlayer:
    def act(self, infoset):
        legal = infoset["legal_actions"]
        pot = infoset.get("current_pot", 0)
        bombs = [a for a in legal if a is not None and move_obj(a).type == "bomb"]
        if bombs and pot < 15 and len(legal) > 1:
            if None in legal:
                return None
            non_bombs = [a for a in legal if a is not None and move_obj(a).type != "bomb"]
            if non_bombs:
                return min(non_bombs, key=lambda a: (move_points(a), len(a), move_obj(a).strength))
        if bombs and pot >= 15:
            return max(bombs, key=lambda a: move_obj(a).strength)
        non_bombs = [a for a in legal if a is not None]
        if not non_bombs:
            return None
        return min(non_bombs, key=lambda a: (move_points(a), len(a), move_obj(a).strength))


class PointDenialPlayer:
    def act(self, infoset):
        legal = infoset["legal_actions"]
        non_pass = [a for a in legal if a is not None]
        if not non_pass:
            return None
        safe = []
        for a in non_pass:
            m = move_obj(a)
            penalty = (
                move_points(a),
                int(m.type == "bomb"),
                int(any(c.rank >= 20 for c in m.cards)),
                max(c.rank for c in m.cards),
                len(m.cards),
            )
            safe.append((penalty, a))
        if None in legal and infoset.get("current_pot", 0) >= 10:
            return None
        return min(safe, key=lambda x: x[0])[1]


class EndgameConserverPlayer:
    def act(self, infoset):
        legal = infoset["legal_actions"]
        non_pass = [a for a in legal if a is not None]
        if not non_pass:
            return None
        finishers = [a for a in non_pass if len(a) == infoset.get("hand_size", len(a))]
        if finishers:
            return min(finishers, key=lambda a: (move_points(a), move_obj(a).strength))
        endgame = infoset.get("deck_size", 0) == 0 or infoset.get("hand_size", 0) <= 3
        scored = []
        for a in non_pass:
            m = move_obj(a)
            max_rank = max(c.rank for c in m.cards)
            conserve_cost = (
                int(endgame and max_rank >= 17),
                int(any(c.rank >= 20 for c in m.cards)),
                int(m.type == "bomb"),
                move_points(a),
                max_rank,
                len(a),
            )
            scored.append((conserve_cost, a))
        if None in legal and infoset.get("current_pot", 0) >= 10 and infoset.get("has_control", 0) == 0:
            return None
        return min(scored, key=lambda x: x[0])[1]


class ControlHoarderPlayer:
    """Patient human style: dump low junk, hoard A/2/jokers, only spend control to
    capture a worthwhile pot; pass rather than burn a high card on a small pot."""
    def act(self, infoset):
        legal = infoset["legal_actions"]
        non_pass = [a for a in legal if a is not None]
        if not non_pass:
            return None
        pot = infoset.get("current_pot", 0)
        leading = infoset.get("hand_type") is None
        if leading:
            # lead the cheapest, lowest, non-point card to probe
            return min(non_pass, key=lambda a: (move_points(a), int(any(c.rank >= 14 for c in move_obj(a).cards)), max(c.rank for c in move_obj(a).cards)))
        # following: only commit a control card if the pot is worth it
        cheap = [a for a in non_pass if not any(c.rank >= 14 for c in move_obj(a).cards) and move_obj(a).type != "bomb"]
        if cheap:
            return min(cheap, key=lambda a: (move_points(a), max(c.rank for c in move_obj(a).cards)))
        if None in legal and pot < 10:
            return None
        return min(non_pass, key=lambda a: (int(move_obj(a).type == "bomb"), max(c.rank for c in move_obj(a).cards), move_points(a)))


class AggressivePointPlayer:
    """Greedy human style: fight hard for any pot that holds points, happily spend
    high cards to win them, and grab the lead."""
    def act(self, infoset):
        legal = infoset["legal_actions"]
        non_pass = [a for a in legal if a is not None]
        if not non_pass:
            return None
        pot = infoset.get("current_pot", 0)
        leading = infoset.get("hand_type") is None
        if leading:
            # lead something mid/strong to pressure; prefer non-point winners
            return max(non_pass, key=lambda a: (move_obj(a).type != "bomb", -move_points(a), max(c.rank for c in move_obj(a).cards)))
        if pot >= 5:
            # contest the pot with the strongest available non-bomb, then bombs
            non_bomb = [a for a in non_pass if move_obj(a).type != "bomb"]
            pool = non_bomb if non_bomb else non_pass
            return max(pool, key=lambda a: (max(c.rank for c in move_obj(a).cards), -move_points(a)))
        if None in legal:
            return None
        return min(non_pass, key=lambda a: (move_points(a), max(c.rank for c in move_obj(a).cards)))


SCRIPTED_BUILDERS: Dict[str, Callable[[], object]] = {
    "trap_bomb": TrapBombPlayer,
    "point_denial": PointDenialPlayer,
    "endgame_conserver": EndgameConserverPlayer,
    "control_hoarder": ControlHoarderPlayer,
    "aggressive_point": AggressivePointPlayer,
}


# ======================
# Model
# ======================


class HistoryBeliefPVNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, history_dim: int, hidden: int = 256, hist_hidden: int = 160):
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
        # Opponent-hand belief heads (supervised by privileged labels). They read
        # the raw context ctx0; their (detached) predictions are then fused back
        # into the decision context so the policy/value condition on the belief.
        self.opp_rank_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, len(RANKS)))
        self.opp_bomb_head = nn.Linear(hidden, 1)
        self.opp_empty1_head = nn.Linear(hidden, 1)
        self.opp_empty2_head = nn.Linear(hidden, 1)
        self.opp_point_head = nn.Linear(hidden, 1)
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
            return torch.zeros((history_batch.size(0), self.hist_hidden), dtype=history_batch.dtype, device=history_batch.device)
        x = self.history_in(history_batch)
        safe_lengths = history_lengths.clamp(min=1).to(torch.int64).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, safe_lengths, batch_first=True, enforce_sorted=False)
        _, h = self.history_rnn(packed)
        hist = h[-1]
        zero_mask = (history_lengths == 0).unsqueeze(1)
        hist = hist.masked_fill(zero_mask, 0.0)
        return hist

    def compute_context(self, state_batch: torch.Tensor, history_batch: torch.Tensor, history_lengths: torch.Tensor):
        s = self.state_net(state_batch)
        h = self.encode_history(history_batch, history_lengths)
        ctx0 = self.context_net(torch.cat([s, h], dim=-1))
        aux = {
            "opp_rank": self.opp_rank_head(ctx0),
            "opp_bomb": self.opp_bomb_head(ctx0).squeeze(-1),
            "opp_empty1": self.opp_empty1_head(ctx0).squeeze(-1),
            "opp_empty2": self.opp_empty2_head(ctx0).squeeze(-1),
            "opp_points": self.opp_point_head(ctx0).squeeze(-1),
        }
        # Detached belief vector -> the supervised aux loss owns the belief heads,
        # while the policy learns to act on a stable predicted belief (consistent
        # between train and inference, since both consume the prediction).
        belief = torch.cat([
            torch.sigmoid(aux["opp_rank"]),
            torch.sigmoid(aux["opp_bomb"]).unsqueeze(-1),
            torch.sigmoid(aux["opp_empty1"]).unsqueeze(-1),
            torch.sigmoid(aux["opp_empty2"]).unsqueeze(-1),
            torch.sigmoid(aux["opp_points"]).unsqueeze(-1),
        ], dim=-1).detach()
        ctx = self.belief_fuse(torch.cat([ctx0, belief], dim=-1))
        return ctx, aux

    def score_actions(self, state_batch: torch.Tensor, history_batch: torch.Tensor, history_lengths: torch.Tensor, action_batch: torch.Tensor):
        ctx, aux = self.compute_context(state_batch, history_batch, history_lengths)
        a = self.action_net(action_batch)
        ctx_exp = ctx.unsqueeze(1).expand(-1, action_batch.shape[1], -1)
        joint = torch.cat([ctx_exp, a], dim=-1)
        logits = self.policy_net(joint).squeeze(-1)
        values = self.value_head(ctx).squeeze(-1)
        return logits, values, aux


# ======================
# PPO storage
# ======================


@dataclass
class Transition:
    state: np.ndarray
    history: np.ndarray
    history_len: int
    legal_action_feats: np.ndarray
    action: int
    log_prob: float
    value: float
    reward: float
    aux_labels: Dict[str, np.ndarray]
    q_values: Optional[np.ndarray]
    context: Dict


# ======================
# Player wrappers
# ======================


class ModelPlayer:
    def __init__(self, model: HistoryBeliefPVNet, device: torch.device, training: bool = False, storage: Optional[List[Transition]] = None):
        self.model = model
        self.device = device
        self.training = training
        self.storage = storage

    def encode(self, infoset):
        state = encode_state(infoset)
        history = encode_history_events(infoset)
        legal_actions = infoset["legal_actions"]
        action_feats = np.stack([encode_move(a, infoset) for a in legal_actions]).astype(np.float32)
        return state, history, legal_actions, action_feats

    def evaluate_infoset(self, infoset):
        state, history, legal_actions, action_feats = self.encode(infoset)
        with torch.no_grad():
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            hist_t = torch.from_numpy(history if len(history) else np.zeros((0, HISTORY_EVENT_DIM), dtype=np.float32)).float().unsqueeze(0).to(self.device)
            hist_len_t = torch.tensor([len(history)], dtype=torch.long, device=self.device)
            acts_t = torch.from_numpy(action_feats).float().unsqueeze(0).to(self.device)
            logits, values, _ = self.model.score_actions(state_t, hist_t, hist_len_t, acts_t)
        return state, history, legal_actions, action_feats, logits[0].cpu().numpy(), float(values[0].item())

    def act(self, infoset, q_values: Optional[np.ndarray] = None):
        state, history, legal_actions, action_feats, logits_np, value = self.evaluate_infoset(infoset)
        logits_t = torch.from_numpy(logits_np)
        dist = torch.distributions.Categorical(logits=logits_t)
        if self.training:
            idx_t = dist.sample()
        else:
            idx_t = torch.argmax(logits_t)
        action_idx = int(idx_t.item())
        log_prob = float(dist.log_prob(idx_t).item())
        move = legal_actions[action_idx]

        if self.training and self.storage is not None:
            bomb_available = any((a is not None and move_obj(a).type == "bomb") for a in legal_actions)
            self.storage.append(Transition(
                state=state,
                history=history,
                history_len=len(history),
                legal_action_feats=action_feats,
                action=action_idx,
                log_prob=log_prob,
                value=value,
                reward=0.0,
                aux_labels={
                    "opp_rank": np.asarray(infoset["opp_rank_counts"], dtype=np.float32) / 4.0,
                    "opp_bomb": np.asarray(infoset["opp_has_bomb"], dtype=np.float32),
                    "opp_empty1": np.asarray(infoset["opp_can_empty_1"], dtype=np.float32),
                    "opp_empty2": np.asarray(infoset["opp_can_empty_2"], dtype=np.float32),
                    # /50 (not /40): a 5-card hand holds up to ~50 points, so /40 pushed
                    # the sigmoid-regressed target past 1.0 and biased the belief head.
                    "opp_points": np.asarray(infoset["opp_point_total"], dtype=np.float32) / 50.0,
                },
                q_values=None if q_values is None else np.asarray(q_values, dtype=np.float32),
                context={
                    "bomb_available": bomb_available,
                    "opp_card_count": infoset.get("opp_card_count", 5),
                    "current_pot": infoset.get("current_pot", 0),
                    "is_endgame": infoset.get("is_endgame", 0),
                    "num_actions": len(legal_actions),
                },
            ))
        return move


# ======================
# League / baseline
# ======================


@dataclass
class OpponentSpec:
    kind: str
    builder: Callable[[HistoryBeliefPVNet, torch.device], object]

    def make_player(self, model: HistoryBeliefPVNet, device: torch.device):
        return self.builder(model, device)


@dataclass
class OpponentEntry:
    id: str
    kind: str  # 'self' | 'random' | 'scripted' | 'snapshot' | 'baseline'
    model: Optional[object] = None
    scripted_name: Optional[str] = None
    wins: float = 0.0   # current policy's (decayed) wins vs this opponent
    games: float = 0.0

    @property
    def kind_tag(self):
        return self.scripted_name if self.kind == "scripted" else self.kind

    def win_rate(self):
        return (self.wins / self.games) if self.games > 0 else None

    def make_player(self, current_model, device):
        if self.kind == "self":
            return ModelPlayer(current_model, device=device, training=False)
        if self.kind == "random":
            return RandomPlayer()
        if self.kind == "scripted":
            return SCRIPTED_BUILDERS[self.scripted_name]()
        return ModelPlayer(self.model, device=device, training=False)  # snapshot / baseline / exploiter


class LeaguePool:
    def __init__(self):
        self.all: List[HistoryBeliefPVNet] = []
        self.recent = deque(maxlen=RECENT_POOL_SIZE)
        self.best: Optional[HistoryBeliefPVNet] = None
        self.baseline: Optional[HistoryBeliefPVNet] = None
        self.last_eval = {
            "baseline_wr": 0.0,
            "pool_wr": 0.0,
            "best_wr": 0.0,
            "frontier_wr": 0.0,
            "scripted_wr": float("nan"),
        }
        # PFSP opponent registry (persistent + snapshots).
        self.entries: Dict[str, OpponentEntry] = {}
        self._snap_counter = 0
        self.version = 0  # bumped whenever opponent *weights* change (for worker sync)
        self.entries["self"] = OpponentEntry("self", "self")
        self.entries["random"] = OpponentEntry("random", "random")
        for name in SCRIPTED_BUILDERS:
            self.entries[f"scripted:{name}"] = OpponentEntry(f"scripted:{name}", "scripted", scripted_name=name)

    def set_baseline(self, model):
        self.baseline = clone_eval_model(model)
        self.entries["baseline"] = OpponentEntry("baseline", "baseline", model=self.baseline)
        self.version += 1

    def _snapshot_entries(self):
        return [e for e in self.entries.values() if e.kind == "snapshot"]

    def add(self, model):
        snap = clone_eval_model(model)
        self.all.append(snap)
        self.recent.append(snap)
        self.best = snap
        self._snap_counter += 1
        sid = f"snap:{self._snap_counter}"
        self.entries[sid] = OpponentEntry(sid, "snapshot", model=snap)
        self.version += 1
        # Evict by *easiness*, not age: keep the snapshots that still challenge us.
        snaps = self._snapshot_entries()
        while len(snaps) > LEAGUE_SIZE:
            easiest = max(snaps, key=lambda e: (e.win_rate() if e.win_rate() is not None else -1.0))
            self.entries.pop(easiest.id, None)
            try:
                self.all.remove(easiest.model)
            except ValueError:
                pass
            snaps = self._snapshot_entries()

    def record_outcome(self, opp_id, win):
        e = self.entries.get(opp_id)
        if e is None:
            return
        e.wins = PFSP_DECAY * e.wins + (1.0 if win else 0.0)
        e.games = PFSP_DECAY * e.games + 1.0

    def _entry_weight(self, e):
        if e.games < PFSP_MIN_GAMES:
            w = 1.0  # optimistic: exercise under-sampled opponents
        else:
            p = e.wins / e.games
            w = max(PFSP_FLOOR, (1.0 - p) ** PFSP_ETA)
        if e.kind == "random":
            w = max(w, RANDOM_MIN_WEIGHT)
        return w

    def sample_entry(self) -> OpponentEntry:
        others = [e for e in self.entries.values() if e.kind != "self"]
        weights = [self._entry_weight(e) for e in others]
        self_entry = self.entries["self"]
        sum_other = sum(weights)
        # Reserve a fixed fraction for vanilla self-play (stability).
        w_self = sum_other * SELF_SHARE / max(1e-9, (1.0 - SELF_SHARE)) if sum_other > 0 else 1.0
        pool_entries = others + [self_entry]
        pool_weights = weights + [w_self]
        total = sum(pool_weights)
        r = random.random() * total
        for e, w in zip(pool_entries, pool_weights):
            r -= w
            if r <= 0:
                return e
        return pool_entries[-1]

    def recent_best(self):
        return self.best if self.best is not None else (self.recent[-1] if self.recent else None)

    def frontier(self):
        items = []
        if self.best is not None:
            items.append(self.best)
        items.extend(list(self.recent)[-FRONTIER_SIZE:])
        return dedup_models(items)

    # ---- payloads for parallel workers ----
    def league_payload(self):
        """Serializable description of every opponent (weights only for models)."""
        payload = {}
        for eid, e in self.entries.items():
            item = {"kind": e.kind, "scripted_name": e.scripted_name}
            if e.model is not None:
                item["state_dict"] = cpu_state_dict(e.model)
            payload[eid] = item
        return self.version, payload

    def stats_payload(self):
        return {eid: (e.wins, e.games) for eid, e in self.entries.items()}

    def load_worker_payload(self, payload, state_dim, action_dim, history_dim):
        self.entries = {}
        for eid, item in payload.items():
            model = None
            if "state_dict" in item:
                model = HistoryBeliefPVNet(state_dim, action_dim, history_dim).to(DEVICE)
                model.load_state_dict(item["state_dict"])
                model.eval()
            self.entries[eid] = OpponentEntry(eid, item["kind"], model=model, scripted_name=item.get("scripted_name"))

    def apply_stats_payload(self, stats):
        for eid, (w, g) in stats.items():
            e = self.entries.get(eid)
            if e is not None:
                e.wins = w
                e.games = g

    def add_exploiter(self, exploiter_model):
        """Add a frozen exploiter snapshot to the PFSP pool. The exploiter's win rate
        vs. the main will be tracked; PFSP will keep sampling it until the main fixes the
        weakness it found. Evicts the easiest existing exploiter if at capacity."""
        snap = clone_eval_model(exploiter_model)
        self._snap_counter += 1
        eid = f"exploiter:{self._snap_counter}"
        self.entries[eid] = OpponentEntry(eid, "exploiter", model=snap)
        self.version += 1
        # Evict oldest exploiter if over cap
        exploiter_entries = [e for e in self.entries.values() if e.kind == "exploiter"]
        while len(exploiter_entries) > EXPLOITER_MAX_IN_LEAGUE:
            # Remove the one the main beats most easily (lowest threat)
            easiest = max(exploiter_entries, key=lambda e: e.win_rate() if e.win_rate() is not None else -1.0)
            self.entries.pop(easiest.id, None)
            exploiter_entries = [e for e in self.entries.values() if e.kind == "exploiter"]
        print(f"  [exploiter] Added {eid} to league. Active exploiters: {len([e for e in self.entries.values() if e.kind=='exploiter'])}", flush=True)


def dedup_models(models):
    seen = set()
    out = []
    for m in models:
        if m is None:
            continue
        ident = id(m)
        if ident not in seen:
            out.append(m)
            seen.add(ident)
    return out


def clone_eval_model(model):
    clone = copy.deepcopy(model).to(DEVICE)
    clone.eval()
    return clone


def cpu_state_dict(model):
    """CPU copy of a model's weights, safe to pickle across processes."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


class ExploiterAgent:
    """A dedicated agent trained ONLY to beat the current main policy.

    Based on the AlphaStar / PSRO exploiter concept. The key distinction from the
    main agent:
      - Opponent is always a frozen snapshot of the current main (not PFSP variety).
      - Reward is pure win/loss (not point margin) -- goal is to win, full stop.
      - Higher LR for fast convergence to exploitative strategies.
      - No auxiliary belief heads, no shaping, no league complexity.
      - When it consistently beats the main, it gets frozen and added to the league,
        forcing the main to learn to defend against its discovered weakness.

    Exploiters auto-find what humans find by playtesting -- risky point-pairs,
    wasted 2s, poor endgame -- without us having to hand-code those as archetypes."""

    def __init__(self, state_dim, action_dim, hidden=256, hist_hidden=160):
        self.model = HistoryBeliefPVNet(state_dim, action_dim, HISTORY_EVENT_DIM,
                                        hidden=hidden, hist_hidden=hist_hidden).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=EXPLOITER_LR)
        self._target: Optional[HistoryBeliefPVNet] = None   # frozen main policy snapshot

    def update_target(self, main_model: HistoryBeliefPVNet):
        """Freeze a copy of the current main as the exploitation target."""
        self._target = clone_eval_model(main_model)

    def _collect(self, n_rollouts):
        """Collect rollouts: exploiter vs. frozen main. Pure win/loss reward."""
        storage, rewards = [], []
        target_player = ModelPlayer(self._target, device=DEVICE, training=False)
        for _ in range(n_rollouts):
            exp_storage: List[Transition] = []
            exp_player = ModelPlayer(self.model, device=DEVICE, training=True, storage=exp_storage)
            seat = random.randint(0, 1)
            players = [None, None]
            players[seat] = exp_player
            players[1 - seat] = target_player
            env = GameEnv(players, verbose=False)
            step_rewards = []
            while not env.done:
                if env.deck.size() == 0 and USE_ENDGAME_SOLVER:
                    # Exact endgame -- exploiter plays optimally here too
                    from endgame import endgame_value_from_env
                    _ = endgame_value_from_env(env, seat)
                    break
                if env.current_player != seat:
                    env.apply_action(players[env.current_player].act(env.get_infoset(env.current_player)))
                    continue
                pre = env.points[seat] - env.points[1 - seat]
                env.apply_action(exp_player.act(env.get_infoset(seat)))
                while not env.done and env.current_player != seat:
                    env.apply_action(players[env.current_player].act(env.get_infoset(env.current_player)))
                step_rewards.append((env.points[seat] - env.points[1 - seat] - pre) / EXPLOITER_WIN_SCALE)
            # Terminal win/loss bonus -- this is the ONLY signal the exploiter cares about
            if exp_storage:
                margin = env.points[seat] - env.points[1 - seat]
                step_rewards[-1] += 1.0 if margin > 0 else -1.0
                for t in exp_storage:
                    t.reward = 0.0
                for i, r in enumerate(step_rewards):
                    if i < len(exp_storage):
                        exp_storage[i].reward = r
                storage.extend(exp_storage)
                rewards.extend(step_rewards)
        return storage, rewards

    def train(self, n_rollouts=EXPLOITER_TRAIN_STEPS):
        if self._target is None:
            return
        storage, rewards = self._collect(n_rollouts)
        if not storage:
            return
        values = [d.value for d in storage]
        advantages, returns = compute_gae(rewards, values)
        # Single PPO epoch (fast exploitation, not stability-focused)
        states_t = torch.tensor(np.array([d.state for d in storage]), dtype=torch.float32, device=DEVICE)
        actions_t = torch.tensor([d.action for d in storage], dtype=torch.long, device=DEVICE)
        old_lp_t = torch.tensor([d.log_prob for d in storage], dtype=torch.float32, device=DEVICE)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        padded_a, mask = pad_action_sets([d.legal_action_feats for d in storage])
        act_t = torch.tensor(padded_a, dtype=torch.float32, device=DEVICE)
        mask_t = torch.tensor(mask, dtype=torch.bool, device=DEVICE)
        padded_h, hlens = pad_histories([d.history for d in storage])
        hist_t = torch.tensor(padded_h, dtype=torch.float32, device=DEVICE)
        hlen_t = torch.tensor(hlens, dtype=torch.long, device=DEVICE)
        idxs = np.arange(len(storage))
        np.random.shuffle(idxs)
        for start in range(0, len(storage), MINIBATCH_SIZE):
            mb = idxs[start:start + MINIBATCH_SIZE]
            logits, vals, _ = self.model.score_actions(states_t[mb], hist_t[mb], hlen_t[mb], act_t[mb])
            logits = logits.masked_fill(~mask_t[mb], -1e9)
            new_lp = torch.log_softmax(logits, dim=-1).gather(1, actions_t[mb].unsqueeze(1)).squeeze(1)
            ratio = torch.exp(new_lp - old_lp_t[mb])
            surr = torch.min(ratio * adv_t[mb], torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * adv_t[mb])
            loss = -surr.mean() + VALUE_COEF * nn.MSELoss()(vals, ret_t[mb])
            self.optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.7)
            self.optimizer.step()

    def win_rate_vs_target(self, n_games=EXPLOITER_EVAL_GAMES):
        """How often the exploiter beats the frozen main target."""
        if self._target is None:
            return 0.0
        wins = 0
        target_player = ModelPlayer(self._target, device=DEVICE, training=False)
        exploiter_player = ModelPlayer(self.model, device=DEVICE, training=False)
        for g in range(n_games):
            seat = g % 2
            players = [None, None]
            players[seat] = exploiter_player
            players[1 - seat] = target_player
            env = GameEnv(players, verbose=False)
            while not env.done:
                env.apply_action(players[env.current_player].act(env.get_infoset(env.current_player)))
            wins += int(env.points[seat] - env.points[1 - seat] > 0)
        return wins / n_games


def make_model_builder(snapshot_model):
    return lambda _current_model, device: ModelPlayer(snapshot_model, device=device, training=False)


def make_random_builder():
    return lambda _current_model, _device: RandomPlayer()


def make_scripted_builder(name: str):
    return lambda _current_model, _device: SCRIPTED_BUILDERS[name]()


def make_baseline_builder(pool: LeaguePool):
    return lambda _current_model, device: ModelPlayer(pool.baseline, device=device, training=False)


def sample_opponent(pool: LeaguePool, model: HistoryBeliefPVNet, ep: int) -> OpponentEntry:
    """PFSP: returns an OpponentEntry whose .make_player / .id drive the rollout."""
    return pool.sample_entry()


# ======================
# Rewards and lookahead
# ======================


def compute_gae(rewards, values, bootstrap=0.0):
    advantages = []
    gae = 0.0
    values = values + [bootstrap]  # bootstrap = exact endgame value when truncated, else 0 (terminal)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + GAMMA * values[t + 1] - values[t]
        gae = delta + GAMMA * LAMBDA * gae
        advantages.insert(0, gae)
    returns = [a + v for a, v in zip(advantages, values[:-1])]
    return advantages, returns


def is_critical_state(infoset):
    legal = infoset["legal_actions"]
    bomb_legal = any((a is not None and move_obj(a).type == "bomb") for a in legal)
    return (
        infoset.get("deck_size", 0) <= 6 or
        infoset.get("current_pot", 0) >= 10 or
        infoset.get("opp_about_to_win", 0) or
        infoset.get("can_empty_hand", 0) or
        infoset.get("hand_size", 0) <= 3 or
        bomb_legal
    )


def replay_prob(ep: int, buffer_size: int):
    if buffer_size < REPLAY_MIN_BUFFER or ep < REPLAY_START_EP:
        return 0.0
    ramp = min(1.0, (ep - REPLAY_START_EP) / 600.0)
    return REPLAY_MAX_PROB * ramp


def value_from_infoset(model: HistoryBeliefPVNet, device: torch.device, infoset):
    state = encode_state(infoset)
    history = encode_history_events(infoset)
    action_feats = np.stack([encode_move(a, infoset) for a in infoset["legal_actions"]]).astype(np.float32)
    with torch.no_grad():
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
        hist_t = torch.from_numpy(history if len(history) else np.zeros((0, HISTORY_EVENT_DIM), dtype=np.float32)).float().unsqueeze(0).to(device)
        hist_len_t = torch.tensor([len(history)], dtype=torch.long, device=device)
        acts_t = torch.from_numpy(action_feats).float().unsqueeze(0).to(device)
        _, values, _ = model.score_actions(state_t, hist_t, hist_len_t, acts_t)
    return float(values[0].item())


def determinize_snapshot(snapshot, model_seat):
    """PIMC determinization: resample the opponent's hand and the deck order from
    the cards the model cannot see, keeping the model's hand, the played cards,
    points and public history fixed. Removes the perfect-information leak in the
    old lookahead (which rolled out from the opponent's *true* hand) and gives the
    sampled rollouts genuine variance."""
    snap = copy.deepcopy(snapshot)
    opp = 1 - model_seat
    hidden = list(snap["hands"][opp]) + list(snap["deck_cards"])
    random.shuffle(hidden)
    n_opp = len(snap["hands"][opp])
    snap["hands"][opp] = hidden[:n_opp]
    snap["deck_cards"] = hidden[n_opp:]
    return snap


def _build_all_cards():
    cards = []
    for r in RANKS:
        if r in (SMALL_JOKER, BIG_JOKER):
            cards.append(Card(r))
        else:
            for s in SUITS:
                cards.append(Card(r, s))
    return cards


ALL_CARDS = _build_all_cards()  # every (rank, suit) is unique in this single deck


ENDGAME_SEARCH_DECK = 8   # at/under this deck size, the search rolls out to terminal
BELIEF_FLOOR = 0.15       # uniform floor mixed into belief-biased determinization


def snapshot_from_infoset(infoset, model_seat, opp_rank_belief=None):
    """Build a *determinized* full-game snapshot from one player's imperfect-info
    view: the opponent's hidden hand and the deck are sampled from the cards the
    player has not seen, consistent with public counts. Enables PIMC search at
    inference (no god-view required).

    If opp_rank_belief (a {rank: prob} estimate from the belief head) is given, the
    opponent's hand is sampled *weighted by that belief* rather than uniformly, so
    the search reasons about the hand we've actually inferred from play -- e.g. it
    won't assume the opponent can beat a pair of 10s if the belief says otherwise."""
    me = model_seat
    opp = 1 - me
    seen = set((c.rank, c.suit or "") for c in infoset["hand"])
    for c in infoset["played_cards"]:
        seen.add((c.rank, c.suit or ""))
    unseen = [c for c in ALL_CARDS if (c.rank, c.suit or "") not in seen]
    n_opp = min(infoset.get("opp_card_count", 0), len(unseen))
    if opp_rank_belief is None or n_opp == 0 or n_opp == len(unseen):
        random.shuffle(unseen)
        opp_hand = unseen[:n_opp]
        deck_cards = unseen[n_opp:]
    else:
        w = np.array([opp_rank_belief.get(c.rank, 0.25) + BELIEF_FLOOR for c in unseen], dtype=np.float64)
        w = w / w.sum()
        idx = np.random.choice(len(unseen), size=n_opp, replace=False, p=w)
        opp_set = set(int(i) for i in idx)
        opp_hand = [unseen[i] for i in idx]
        deck_cards = [unseen[i] for i in range(len(unseen)) if i not in opp_set]
    played = list(infoset["played_cards"])
    prc = {r: 0 for r in RANKS}
    for c in played:
        prc[c.rank] += 1
    return {
        "seed": None,
        "deck_cards": deck_cards,
        "hands": {me: list(infoset["hand"]), opp: opp_hand},
        "points": dict(infoset["points"]),
        "done": False,
        "played_cards": played,
        "played_rank_counts": prc,
        "last_hand_points": infoset.get("last_hand_points", 0),
        "last_hand_winner": infoset.get("last_hand_winner"),
        "face_up": {},
        "current_player": me,
        "hand_type": infoset.get("hand_type"),
        "last_move": infoset.get("last_move"),
        "last_player": infoset.get("last_player"),
        "last_action_was_pass": infoset.get("last_action_was_pass", False),
        "pass_count": infoset.get("pass_count", 0),
        "current_pot": infoset.get("current_pot", 0),
        "public_history": list(infoset.get("public_history", [])),
        "history_maxlen": HISTORY_MAXLEN,
    }


class SearchModelPlayer:
    """Inference-time PIMC search: for each legal action, average its value over
    sampled determinizations of the hidden state (short rollout vs the net as the
    opponent model, then bootstrap with the value head). Picks the best action.

    This is what makes the deployed bot tactically sharp -- it will not spend a
    control card to win nothing once it can *see* (via sampling) that the points
    on the table do not justify it. Inference/eval only; too slow for training."""

    wants_concrete_same_rank_choices = False

    def __init__(self, model, device, determinizations: int = 20, horizon: int = 10):
        self.model = model
        self.device = device
        self.K = determinizations
        self.H = horizon

    def _players(self, me):
        p = [None, None]
        p[me] = ModelPlayer(self.model, device=self.device, training=False)
        p[1 - me] = ModelPlayer(self.model, device=self.device, training=False)
        return p

    def _opp_rank_belief(self, infoset):
        """One forward pass -> the belief head's per-rank estimate of the opponent's
        hand, used to bias determinization sampling."""
        state = encode_state(infoset)
        history = encode_history_events(infoset)
        with torch.no_grad():
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            hist_arr = history if len(history) else np.zeros((0, HISTORY_EVENT_DIM), dtype=np.float32)
            hist_t = torch.from_numpy(hist_arr).float().unsqueeze(0).to(self.device)
            hist_len_t = torch.tensor([len(history)], dtype=torch.long, device=self.device)
            _, aux = self.model.compute_context(state_t, hist_t, hist_len_t)
            probs = torch.sigmoid(aux["opp_rank"])[0].cpu().numpy()
        return {RANKS[i]: float(probs[i]) for i in range(len(RANKS))}

    def act(self, infoset):
        legal = infoset["legal_actions"]
        if len(legal) <= 1:
            return legal[0] if legal else None
        me = infoset["player_index"]
        belief = self._opp_rank_belief(infoset)
        # In the endgame (deck nearly gone) there is little hidden information and the
        # game is a short deterministic race -- so roll all the way to terminal and let
        # the true outcome (run-out + the +20) drive the choice, instead of trusting a
        # myopic value bootstrap. This is what fixes weak endgame / tempo play.
        endgame = infoset.get("deck_size", 99) <= ENDGAME_SEARCH_DECK
        horizon_cap = 64 if endgame else self.H
        snaps = [snapshot_from_infoset(infoset, me, belief) for _ in range(self.K)]  # common random numbers
        best_a, best_q = legal[0], float("-inf")
        for a in legal:
            total = 0.0
            for snap in snaps:
                players = self._players(me)
                env = GameEnv.from_snapshot(snap, players, verbose=False)
                m0 = env.points[me] - env.points[1 - me]
                env.apply_action(a)
                depth = 0
                while not env.done and depth < horizon_cap:
                    cur = env.current_player
                    if cur == me and not endgame:
                        break  # mid-game: stop at our next decision and bootstrap with V
                    env.apply_action(players[cur].act(env.get_infoset(cur)))
                    depth += 1
                m1 = env.points[me] - env.points[1 - me]
                bootstrap = 0.0
                if not env.done and env.current_player == me:
                    bootstrap = POINT_SCALE * value_from_infoset(self.model, self.device, env.get_infoset(me))
                total += (m1 - m0 + bootstrap)
            q = total / len(snaps)
            if q > best_q:
                best_q, best_a = q, a
        return best_a


def _policy_value(model, device, infoset):
    """One forward pass -> (softmax priors over legal actions, value)."""
    state = encode_state(infoset)
    history = encode_history_events(infoset)
    af = np.stack([encode_move(a, infoset) for a in infoset["legal_actions"]]).astype(np.float32)
    with torch.no_grad():
        st = torch.from_numpy(state).float().unsqueeze(0).to(device)
        hist_arr = history if len(history) else np.zeros((0, HISTORY_EVENT_DIM), dtype=np.float32)
        ht = torch.from_numpy(hist_arr).float().unsqueeze(0).to(device)
        hl = torch.tensor([len(history)], dtype=torch.long, device=device)
        at = torch.from_numpy(af).float().unsqueeze(0).to(device)
        logits, values, _ = model.score_actions(st, ht, hl, at)
        priors = torch.softmax(logits[0], dim=-1).cpu().numpy()
        val = float(values[0].item())
    return priors, val


def _action_key(move):
    """Canonical, determinization-invariant key for a move (suit-agnostic)."""
    if move is None:
        return ("pass",)
    m = move if isinstance(move, Move) else Move(move)
    return (m.type, m.strength, m.length, tuple(sorted(c.rank for c in m.cards)))


class _ISNode:
    __slots__ = ("N", "W", "P", "expanded")

    def __init__(self):
        self.N = {}
        self.W = {}
        self.P = {}
        self.expanded = False


class ISMCTSPlayer:
    """Information-Set MCTS: a single tree over the searching player's information sets,
    with a fresh belief-sampled determinization each iteration (the opponent's hidden
    actions vary across determinizations, but the tree statistics are shared at the
    info-set level -- this is what avoids the PIMC strategy-fusion pathology). PUCT uses
    the policy net as the action prior; leaves are evaluated by the value net, and any
    deck-empty node is evaluated EXACTLY by the endgame solver."""

    wants_concrete_same_rank_choices = False

    def __init__(self, model, device, iterations: int = 160, c_puct: float = 1.5,
                 depth_limit: int = 14, use_endgame: bool = True):
        self.model = model
        self.device = device
        self.iters = iterations
        self.c = c_puct
        self.depth_limit = depth_limit
        self.use_endgame = use_endgame

    def _players(self, me):
        p = [None, None]
        p[me] = ModelPlayer(self.model, device=self.device, training=False)
        p[1 - me] = ModelPlayer(self.model, device=self.device, training=False)
        return p

    def _opp_belief(self, infoset):
        state = encode_state(infoset)
        history = encode_history_events(infoset)
        with torch.no_grad():
            st = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            hist_arr = history if len(history) else np.zeros((0, HISTORY_EVENT_DIM), dtype=np.float32)
            ht = torch.from_numpy(hist_arr).float().unsqueeze(0).to(self.device)
            hl = torch.tensor([len(history)], dtype=torch.long, device=self.device)
            _, aux = self.model.compute_context(st, ht, hl)
            probs = torch.sigmoid(aux["opp_rank"])[0].cpu().numpy()
        return {RANKS[i]: float(probs[i]) for i in range(len(RANKS))}

    def act(self, infoset):
        legal = infoset["legal_actions"]
        if len(legal) <= 1:
            return legal[0] if legal else None
        me = infoset["player_index"]
        belief = self._opp_belief(infoset)
        tree = {}
        for _ in range(self.iters):
            snap = snapshot_from_infoset(infoset, me, belief)
            env = GameEnv.from_snapshot(snap, self._players(me), verbose=False)
            self._simulate(env, me, tree, (), 0)
        root = tree.get(())
        if root is None or not root.N:
            return legal[0]
        # most-visited action at the root, mapped back to a concrete legal move
        best_key = max(root.N, key=lambda k: root.N[k])
        for a in legal:
            if _action_key(a) == best_key:
                return a
        return legal[0]

    def _simulate(self, env, me, tree, path, depth):
        if env.done:
            return env.points[me] - env.points[1 - me]
        if self.use_endgame and env.deck.size() == 0:
            return (env.points[me] - env.points[1 - me]) + endgame_value_from_env(env, me)
        cur = env.current_player
        info = env.get_infoset(cur)
        legal = info["legal_actions"]
        node = tree.get(path)
        if node is None or not node.expanded:
            if node is None:
                node = _ISNode()
                tree[path] = node
            priors, val = _policy_value(self.model, self.device, info)
            for a, p in zip(legal, priors):
                node.P[_action_key(a)] = float(p)
            node.expanded = True
            return (env.points[me] - env.points[1 - me]) + (val * POINT_SCALE if cur == me else -val * POINT_SCALE)
        if depth >= self.depth_limit:
            _, val = _policy_value(self.model, self.device, info)
            return (env.points[me] - env.points[1 - me]) + (val * POINT_SCALE if cur == me else -val * POINT_SCALE)
        total_n = sum(node.N.get(_action_key(a), 0) for a in legal) + 1
        best_score, best_a, best_k = None, legal[0], _action_key(legal[0])
        for a in legal:
            k = _action_key(a)
            n = node.N.get(k, 0)
            q_me = (node.W.get(k, 0.0) / n) if n > 0 else 0.0
            q_cur = q_me if cur == me else -q_me
            u = self.c * node.P.get(k, 1e-3) * math.sqrt(total_n) / (1 + n)
            score = q_cur + u
            if best_score is None or score > best_score:
                best_score, best_a, best_k = score, a, k
        env.apply_action(best_a)
        val = self._simulate(env, me, tree, path + (best_k,), depth + 1)
        node.N[best_k] = node.N.get(best_k, 0) + 1
        node.W[best_k] = node.W.get(best_k, 0.0) + val
        return val


def estimate_action_values(snapshot, model_seat, model: HistoryBeliefPVNet, opponent_spec, infoset, device: torch.device) -> Optional[np.ndarray]:
    legal = infoset["legal_actions"]
    if len(legal) > LOOKAHEAD_MAX_ACTIONS:
        return None
    q_vals = []
    for action in legal:
        sims = []
        for _ in range(LOOKAHEAD_ROLLOUTS):
            players = [None, None]
            players[model_seat] = ModelPlayer(model, device=device, training=False)
            players[1 - model_seat] = opponent_spec.make_player(model, device)
            env = GameEnv.from_snapshot(determinize_snapshot(snapshot, model_seat), players, verbose=False)
            margin_before = env.points[model_seat] - env.points[1 - model_seat]
            env.apply_action(action)
            steps = 0
            while not env.done and steps < LOOKAHEAD_HORIZON:
                cur = env.current_player
                info = env.get_infoset(cur)
                act = players[cur].act(info)
                env.apply_action(act)
                steps += 1
            margin_after = env.points[model_seat] - env.points[1 - model_seat]
            value_bonus = 0.0
            if not env.done and env.current_player == model_seat:
                value_bonus = POINT_SCALE * value_from_infoset(model, device, env.get_infoset(model_seat))
            sims.append((margin_after - margin_before + value_bonus) / POINT_SCALE)
        q_vals.append(float(np.mean(sims)))
    return np.asarray(q_vals, dtype=np.float32)


# ======================
# Rollout collection
# ======================


def maybe_capture_replay(env: GameEnv, buffer: Deque[dict]):
    info = env.get_infoset(env.current_player)
    if is_critical_state(info):
        buffer.append(env.snapshot())


def effective_shaping_beta(episode):
    """Anneal the control-economy shaping to 0 so the converged policy is trained on the
    pure points/win objective. PBRS never changes the optimum; this just removes any
    residual finite-training bias toward hoarding control cards."""
    if SHAPING_ANNEAL_EPISODES <= 0:
        return SHAPING_BETA
    return SHAPING_BETA * max(0.0, 1.0 - episode / float(SHAPING_ANNEAL_EPISODES))


def collect_rollout(model, pool, episode, replay_buffer: Optional[Deque[dict]] = None):
    storage: List[Transition] = []
    rewards: List[float] = []
    phis: List[float] = []
    beta_eff = effective_shaping_beta(episode)
    stats = Counter()
    model_player = ModelPlayer(model, device=DEVICE, training=True, storage=storage)
    opp_entry = pool.sample_entry()
    model_seat = random.randint(0, 1)
    players = [None, None]
    players[model_seat] = model_player
    players[1 - model_seat] = opp_entry.make_player(model, DEVICE)

    use_replay = False
    if replay_buffer is not None and random.random() < replay_prob(episode, len(replay_buffer)):
        env = GameEnv.from_snapshot(copy.deepcopy(random.choice(list(replay_buffer))), players, verbose=False)
        use_replay = True
    else:
        env = GameEnv(players, verbose=False)

    stats[opp_entry.kind_tag] += 1
    stats[f"seat_{model_seat}"] += 1
    stats["replay"] += int(use_replay)
    total_model_decisions = 0
    truncated = False
    endgame_bootstrap = 0.0

    while not env.done:
        if replay_buffer is not None:
            maybe_capture_replay(env, replay_buffer)

        if env.current_player != model_seat:
            infoset = env.get_infoset(env.current_player)
            act = players[env.current_player].act(infoset)
            env.apply_action(act)
            continue

        # Deck empty -> perfect information. Truncate the rollout here and bootstrap with
        # the EXACT endgame value instead of letting the policy fumble the run-out.
        if USE_ENDGAME_SOLVER and env.deck.size() == 0:
            endgame_bootstrap = endgame_value_from_env(env, model_seat)  # future margin (points)
            truncated = True
            stats["endgame_solved"] += 1
            break

        infoset = env.get_infoset(model_seat)
        phis.append(scaled_potential(infoset["hand"], beta_eff, POINT_SCALE))
        q_values = None
        if USE_LOOKAHEAD and episode >= LOOKAHEAD_START_EP and is_critical_state(infoset):
            q_values = estimate_action_values(env.snapshot(), model_seat, model, opp_entry, infoset, DEVICE)
            if q_values is not None:
                stats["lookahead_targets"] += 1

        pre_margin = env.points[model_seat] - env.points[1 - model_seat]
        act = model_player.act(infoset, q_values=q_values)
        env.apply_action(act)
        reward_acc = env.points[model_seat] - env.points[1 - model_seat] - pre_margin

        while not env.done and env.current_player != model_seat:
            opp_info = env.get_infoset(env.current_player)
            opp_act = players[env.current_player].act(opp_info)
            pre_margin_inner = env.points[model_seat] - env.points[1 - model_seat]
            env.apply_action(opp_act)
            reward_acc += env.points[model_seat] - env.points[1 - model_seat] - pre_margin_inner

        storage[-1].reward = reward_acc / POINT_SCALE
        rewards.append(storage[-1].reward)
        total_model_decisions += 1
        stats["decisions"] += 1
        if storage[-1].context.get("bomb_available", False):
            stats["bomb_opportunities"] += 1
        if act is None:
            stats["passes"] += 1
        else:
            mv = move_obj(act)
            if mv.type == "bomb":
                stats["bombs"] += 1

    if beta_eff > 0.0:
        apply_shaping(rewards, phis, GAMMA)

    if truncated:
        # Game decided by the exact endgame value; eventual margin = current + V*.
        pre = env.points[model_seat] - env.points[1 - model_seat]
        final_margin = pre + endgame_bootstrap
        bootstrap = endgame_bootstrap / POINT_SCALE
        if WIN_BONUS > 0.0:
            bootstrap += (1.0 if final_margin > 0 else -1.0) * WIN_BONUS / POINT_SCALE
        outcome = (opp_entry.id, int(final_margin > 0))
    else:
        margin = env.points[model_seat] - env.points[1 - model_seat]
        if WIN_BONUS > 0.0 and rewards:
            rewards[-1] += (1.0 if margin > 0 else -1.0) * WIN_BONUS / POINT_SCALE
        bootstrap = 0.0
        outcome = (opp_entry.id, int(margin > 0))

    values = [d.value for d in storage]
    advantages, returns = compute_gae(rewards, values, bootstrap)
    return storage, advantages, returns, stats, outcome


# ======================
# Batching / training
# ======================


def pad_action_sets(action_sets: List[np.ndarray]):
    batch = len(action_sets)
    max_actions = max(arr.shape[0] for arr in action_sets)
    action_dim = action_sets[0].shape[1]
    padded = np.zeros((batch, max_actions, action_dim), dtype=np.float32)
    mask = np.zeros((batch, max_actions), dtype=np.bool_)
    for i, arr in enumerate(action_sets):
        n = arr.shape[0]
        padded[i, :n] = arr
        mask[i, :n] = True
    return padded, mask


def pad_histories(histories: List[np.ndarray]):
    batch = len(histories)
    max_len = max((h.shape[0] for h in histories), default=0)
    padded = np.zeros((batch, max_len, HISTORY_EVENT_DIM), dtype=np.float32)
    lengths = np.zeros((batch,), dtype=np.int64)
    for i, h in enumerate(histories):
        lengths[i] = h.shape[0]
        if h.shape[0] > 0:
            padded[i, :h.shape[0]] = h
    return padded, lengths


def pad_q_values(q_values: List[Optional[np.ndarray]], legal_action_feats: List[np.ndarray]):
    """
    Pad lookahead targets onto the same action axis used by the policy logits.

    The rare late-training crash came from padding q-values independently:
    - logits are padded to the batch max number of legal actions
    - q targets were padded to the batch max q-vector length

    Once lookahead turns on, a minibatch can contain an 8-action transition
    whose q_values are None (or absent) and a 7-action transition with q_values.
    That makes logits.shape[1] == 8 but q_mask.shape[1] == 7 and crashes in
    `logits.masked_fill(~mb_q_mask, ...)`.

    We therefore always pad q-values to the action-axis width. We also require
    each q-vector to match the stored legal action count for that transition; a
    misaligned q-vector is dropped for that sample instead of crashing training.
    """
    batch = len(q_values)
    max_len = max((arr.shape[0] for arr in legal_action_feats), default=1)
    padded = np.zeros((batch, max_len), dtype=np.float32)
    mask = np.zeros((batch, max_len), dtype=np.bool_)
    dropped = 0

    for i, (q, action_arr) in enumerate(zip(q_values, legal_action_feats)):
        if q is None:
            continue

        expected_len = int(action_arr.shape[0])
        actual_len = int(len(q))

        if actual_len != expected_len:
            dropped += 1
            continue

        padded[i, :expected_len] = q
        mask[i, :expected_len] = True

    return padded, mask, dropped


def train_step(model, optimizer, batch, entropy_coef):
    states, histories, legal_action_feats, actions, old_log_probs, returns, advantages, aux_labels, q_values, old_values = batch

    states_t = torch.tensor(np.array(states), dtype=torch.float32, device=DEVICE)
    actions_t = torch.tensor(actions, dtype=torch.long, device=DEVICE)
    old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32, device=DEVICE)
    old_values_t = torch.tensor(old_values, dtype=torch.float32, device=DEVICE)
    returns_t = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
    advantages_t = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    padded_actions, action_mask = pad_action_sets(legal_action_feats)
    action_t = torch.tensor(padded_actions, dtype=torch.float32, device=DEVICE)
    mask_t = torch.tensor(action_mask, dtype=torch.bool, device=DEVICE)

    padded_hist, hist_lens = pad_histories(histories)
    hist_t = torch.tensor(padded_hist, dtype=torch.float32, device=DEVICE)
    hist_lens_t = torch.tensor(hist_lens, dtype=torch.long, device=DEVICE)

    opp_rank_t = torch.tensor(np.stack([a["opp_rank"] for a in aux_labels]), dtype=torch.float32, device=DEVICE)
    opp_bomb_t = torch.tensor(np.array([a["opp_bomb"] for a in aux_labels]), dtype=torch.float32, device=DEVICE)
    opp_empty1_t = torch.tensor(np.array([a["opp_empty1"] for a in aux_labels]), dtype=torch.float32, device=DEVICE)
    opp_empty2_t = torch.tensor(np.array([a["opp_empty2"] for a in aux_labels]), dtype=torch.float32, device=DEVICE)
    opp_points_t = torch.tensor(np.array([a["opp_points"] for a in aux_labels]), dtype=torch.float32, device=DEVICE)

    padded_q, q_mask_arr, dropped_q = pad_q_values(q_values, legal_action_feats)
    q_t = torch.tensor(padded_q, dtype=torch.float32, device=DEVICE)
    q_mask_t = torch.tensor(q_mask_arr, dtype=torch.bool, device=DEVICE)
    if dropped_q > 0:
        print(f"[warn] dropped {dropped_q} misaligned lookahead target vector(s) during batch assembly.", flush=True)

    n = states_t.shape[0]
    idxs = np.arange(n)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    kl_running = []
    for _ in range(EPOCHS):
        np.random.shuffle(idxs)
        epoch_kls = []
        for start in range(0, n, MINIBATCH_SIZE):
            mb = idxs[start:start + MINIBATCH_SIZE]
            logits, values, aux = model.score_actions(states_t[mb], hist_t[mb], hist_lens_t[mb], action_t[mb])
            logits = logits.masked_fill(~mask_t[mb], -1e9)
            log_probs = torch.log_softmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)
            new_log_probs = log_probs.gather(1, actions_t[mb].unsqueeze(1)).squeeze(1)
            entropy = -(probs * log_probs).masked_fill(~mask_t[mb], 0.0).sum(dim=-1).mean()

            log_ratio = new_log_probs - old_log_probs_t[mb]
            ratio = torch.exp(log_ratio)
            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean().item()  # Schulman k3, >= 0
                epoch_kls.append(approx_kl)
            surr1 = ratio * advantages_t[mb]
            surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages_t[mb]
            policy_loss = -torch.min(surr1, surr2).mean()
            # clipped value loss: trust-region on the value head so it can't lurch.
            v_clipped = old_values_t[mb] + torch.clamp(values - old_values_t[mb], -VALUE_CLIP, VALUE_CLIP)
            value_loss = torch.max((values - returns_t[mb]) ** 2, (v_clipped - returns_t[mb]) ** 2).mean()

            aux_loss = mse(torch.sigmoid(aux["opp_rank"]), opp_rank_t[mb])
            aux_loss = aux_loss + bce(aux["opp_bomb"], opp_bomb_t[mb])
            aux_loss = aux_loss + bce(aux["opp_empty1"], opp_empty1_t[mb])
            aux_loss = aux_loss + bce(aux["opp_empty2"], opp_empty2_t[mb])
            aux_loss = aux_loss + mse(torch.sigmoid(aux["opp_points"]), opp_points_t[mb])

            lookahead_loss = torch.tensor(0.0, device=DEVICE)
            mb_q_mask = q_mask_t[mb]
            if mb_q_mask.any():
                q_logits = logits.masked_fill(~mb_q_mask, -1e9)
                q_targets = torch.softmax(q_t[mb] / LOOKAHEAD_TEMP, dim=-1)
                q_targets = q_targets.masked_fill(~mb_q_mask, 0.0)
                q_targets = q_targets / (q_targets.sum(dim=-1, keepdim=True) + 1e-8)
                lookahead_loss = -(q_targets * torch.log_softmax(q_logits, dim=-1)).masked_fill(~mb_q_mask, 0.0).sum(dim=-1)
                valid = mb_q_mask.any(dim=-1)
                lookahead_loss = lookahead_loss[valid].mean() if valid.any() else torch.tensor(0.0, device=DEVICE)

            loss = policy_loss + VALUE_COEF * value_loss + AUX_COEF * aux_loss + LOOKAHEAD_COEF * lookahead_loss - entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
            optimizer.step()

        epoch_kl = float(np.mean(epoch_kls)) if epoch_kls else 0.0
        kl_running.append(epoch_kl)
        if epoch_kl > KL_STOP_FACTOR * TARGET_KL:
            break  # policy moved enough this batch; stop before it overshoots

    return float(np.mean(kl_running)) if kl_running else 0.0


# ======================
# Evaluation
# ======================


def play_match(model, opponent_spec: OpponentSpec, games=1):
    wins = 0
    point_diff = 0.0
    for _ in range(games):
        model_seat = random.randint(0, 1)
        players = [None, None]
        players[model_seat] = ModelPlayer(model, device=DEVICE, training=False)
        players[1 - model_seat] = opponent_spec.make_player(model, DEVICE)
        env = GameEnv(players, verbose=False)
        while not env.done:
            infoset = env.get_infoset(env.current_player)
            act = players[env.current_player].act(infoset)
            env.apply_action(act)
        margin = env.points[model_seat] - env.points[1 - model_seat]
        wins += int(margin > 0)
        point_diff += margin
    return wins / games, point_diff / games


def evaluate_random(model, games=RANDOM_EVAL_GAMES):
    return play_match(model, OpponentSpec("random", make_random_builder()), games)[0]


def evaluate_opponents(model, opponent_specs: List[OpponentSpec], games_per_opp: int):
    if not opponent_specs:
        return 0.0
    total = 0
    wins = 0
    for spec in opponent_specs:
        wr, _ = play_match(model, spec, games=games_per_opp)
        wins += wr * games_per_opp
        total += games_per_opp
    return wins / max(1, total)


def refresh_pool_stats(model, pool: LeaguePool):
    baseline_wr = evaluate_opponents(model, [OpponentSpec("baseline", make_baseline_builder(pool))], BASELINE_EVAL_GAMES) if pool.baseline is not None else 0.0
    pool_specs = [OpponentSpec("recent", make_model_builder(m)) for m in list(pool.recent)]
    frontier_specs = [OpponentSpec("frontier", make_model_builder(m)) for m in pool.frontier()]
    best_specs = [OpponentSpec("best", make_model_builder(pool.recent_best()))] if pool.recent_best() is not None else []
    scripted_specs = [OpponentSpec(name, make_scripted_builder(name)) for name in SCRIPTED_BUILDERS]
    stats = {
        "baseline_wr": baseline_wr,
        "pool_wr": evaluate_opponents(model, pool_specs, POOL_EVAL_GAMES) if pool_specs else 0.0,
        "best_wr": evaluate_opponents(model, best_specs, RECENT_BEST_GAMES) if best_specs else 0.0,
        "frontier_wr": evaluate_opponents(model, frontier_specs, FRONTIER_EVAL_GAMES) if frontier_specs else 0.0,
        "scripted_wr": evaluate_opponents(model, scripted_specs, SCRIPTED_EVAL_GAMES),
    }
    pool.last_eval = stats
    return stats


def should_add_snapshot(model, pool: LeaguePool):
    """Add a snapshot (freeze the current main as an opponent) when it's strong enough
    on the external anchors. Snapshots are the league's implicit exploiters: once main
    beats a version, that version stays in the league as a challenger via PFSP weighting.
    Lowered thresholds to keep the league fresh and prevent locked equilibria."""
    if not pool.all:
        return True
    stats = refresh_pool_stats(model, pool)
    # Relaxed: baseline 0.54→0.50, pool/frontier 0.51→0.48, scripted 0.45→0.40.
    # This keeps snapshots flowing in so the league stays diverse and doesn't freeze.
    return (
        stats["baseline_wr"] >= 0.50 and
        stats["pool_wr"] >= 0.48 and
        stats["frontier_wr"] >= 0.48 and
        stats["scripted_wr"] >= 0.40
    )


def composite_eval_score(random_wr: float, eval_stats: dict):
    # Weight the discriminative external benchmark (diverse scripted archetypes)
    # most. Random/baseline saturate near 1.0 and pool/frontier sit near the 0.5
    # self-play equilibrium, so they carry little model-selection signal late.
    return (
        0.10 * random_wr +
        0.20 * eval_stats.get("baseline_wr", 0.0) +
        0.15 * eval_stats.get("pool_wr", 0.0) +
        0.15 * eval_stats.get("frontier_wr", 0.0) +
        0.40 * eval_stats.get("scripted_wr", 0.0)
    )


# ======================
# Checkpoints / plotting
# ======================


def save_checkpoint(model, state_dim: int, action_dim: int, episode: int, path: str):
    torch.save({
        "arch": "history_belief_ppo_v3",
        "model_state_dict": model.state_dict(),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "history_dim": HISTORY_EVENT_DIM,
        "hidden": getattr(model, "hidden", 256),
        "hist_hidden": getattr(model, "hist_hidden", 160),
        "episode": episode,
    }, path)


def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location=DEVICE)
    if ckpt.get("arch") != "history_belief_ppo_v3":
        raise ValueError(
            f"Checkpoint arch {ckpt.get('arch')!r} != 'history_belief_ppo_v3'. "
            "Older checkpoints (different feature dims) are not loadable; retrain."
        )
    model = HistoryBeliefPVNet(ckpt["state_dim"], ckpt["action_dim"], ckpt.get("history_dim", HISTORY_EVENT_DIM), hidden=ckpt.get("hidden", 256), hist_hidden=ckpt.get("hist_hidden", 160)).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def maybe_load_baseline(path: Optional[str]):
    if path is None:
        return None, None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Baseline checkpoint not found: {path}")
    try:
        model, ckpt = load_checkpoint(path)
    except ValueError:
        print(f"Warning: ignoring incompatible baseline checkpoint at {path}.", flush=True)
        return None, None
    meta = {"path": path, "episode": ckpt.get("episode", "unknown")}
    return model, meta


def find_default_baseline(path: Optional[str]):
    if path is not None:
        return path
    if os.path.exists(DEFAULT_BASELINE_CHECKPOINT):
        return DEFAULT_BASELINE_CHECKPOINT
    return None


def make_perf_history():
    return {
        "episode": [],
        "random_wr": [],
        "baseline_wr": [],
        "pool_wr": [],
        "frontier_wr": [],
        "scripted_wr": [],
    }


def record_perf_point(history, episode: int, random_wr: float, eval_stats: dict):
    history["episode"].append(int(episode))
    history["random_wr"].append(float(random_wr))
    history["baseline_wr"].append(float(eval_stats["baseline_wr"]))
    history["pool_wr"].append(float(eval_stats["pool_wr"]))
    history["frontier_wr"].append(float(eval_stats["frontier_wr"]))
    history["scripted_wr"].append(float(eval_stats["scripted_wr"]))


def save_perf_csv(history, path: str = PERF_CSV_PATH):
    fields = ["episode", "random_wr", "baseline_wr", "pool_wr", "frontier_wr", "scripted_wr"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        n = len(history["episode"])
        for i in range(n):
            writer.writerow([history[k][i] for k in fields])


def save_perf_plot(history, path: str = PERF_PNG_PATH):
    if not history["episode"]:
        return
    import matplotlib.pyplot as plt
    episodes = np.asarray(history["episode"], dtype=np.int32)
    plt.figure(figsize=(10, 6))
    for key, label in [
        ("random_wr", "vs random"),
        ("baseline_wr", "vs baseline"),
        ("pool_wr", "vs recent pool"),
        ("frontier_wr", "vs frontier"),
        ("scripted_wr", "vs scripted"),
    ]:
        plt.plot(episodes, np.asarray(history[key], dtype=np.float32), linewidth=2.0, label=label)
    plt.xlabel("Episode")
    plt.ylabel("Win rate")
    plt.title("Policy performance over training")
    plt.ylim(0.0, 1.0)
    if len(episodes) > 1:
        plt.xlim(int(episodes.min()), int(episodes.max()))
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


# ======================
# Parallel actor-learner
# ======================


def _apply_worker_globals(config):
    """Set the module globals a spawned worker needs (it re-imports this module)."""
    global DEVICE, SHAPING_BETA, WIN_BONUS, USE_LOOKAHEAD, USE_ENDGAME_SOLVER
    DEVICE = torch.device("cpu")
    SHAPING_BETA = config["shaping_beta"]
    WIN_BONUS = config["win_bonus"]
    USE_LOOKAHEAD = config["use_lookahead"]
    USE_ENDGAME_SOLVER = config["use_endgame_solver"]
    try:
        torch.set_num_threads(1)        # one core per worker -> no BLAS oversubscription
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _actor_worker(worker_id, conn, config):
    """Spawned process: receive latest weights, return on-policy rollout transitions."""
    _apply_worker_globals(config)
    seed = (config["seed"] + 100003 * (worker_id + 1)) % (2 ** 31 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    sd, ad, hd = config["state_dim"], config["action_dim"], config["history_dim"]
    model = HistoryBeliefPVNet(sd, ad, hd, hidden=config["hidden"], hist_hidden=config["hist_hidden"]).to(DEVICE)
    pool = LeaguePool()
    league_version = -1
    local_replay: Optional[Deque[dict]] = deque(maxlen=REPLAY_BUFFER_SIZE) if config["use_replay"] else None
    try:
        while True:
            msg = conn.recv()
            if msg.get("cmd") == "stop":
                break
            model.load_state_dict(msg["model_state"])
            model.eval()
            if msg.get("league") is not None and msg["league_version"] != league_version:
                pool.load_worker_payload(msg["league"], sd, ad, hd)
                league_version = msg["league_version"]
            pool.apply_stats_payload(msg["league_stats"])
            episode = msg["episode"]
            storage, adv, ret, outcomes = [], [], [], []
            stats = Counter()
            for _ in range(msg["num_rollouts"]):
                s, a, r, st, oc = collect_rollout(model, pool, episode, replay_buffer=local_replay)
                storage += s
                adv += a
                ret += r
                stats.update(st)
                outcomes.append(oc)
            conn.send({"storage": storage, "adv": adv, "ret": ret, "stats": stats, "outcomes": outcomes})
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        conn.close()


class ParallelCollector:
    """Synchronous data-parallel actors: same on-policy PPO, just a bigger,
    decorrelated batch gathered across processes each iteration."""

    def __init__(self, num_workers, config):
        ctx = mp.get_context("spawn")
        self.num_workers = num_workers
        self.conns = []
        self.procs = []
        self._sent_version = -1
        for wid in range(num_workers):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_actor_worker, args=(wid, child, config), daemon=True)
            p.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(p)

    def _split(self, total):
        base, rem = divmod(total, self.num_workers)
        return [base + (1 if i < rem else 0) for i in range(self.num_workers)]

    def collect(self, model, pool, episode, rollouts_per_batch):
        version, league = pool.league_payload()
        send_league = version != self._sent_version
        stats_p = pool.stats_payload()
        model_state = cpu_state_dict(model)
        counts = self._split(rollouts_per_batch)
        for conn, k in zip(self.conns, counts):
            conn.send({
                "model_state": model_state,
                "league_version": version,
                "league": league if send_league else None,
                "league_stats": stats_p,
                "episode": episode,
                "num_rollouts": k,
            })
        self._sent_version = version
        storage, adv, ret, outcomes = [], [], [], []
        stats = Counter()
        for conn in self.conns:
            res = conn.recv()
            storage += res["storage"]
            adv += res["adv"]
            ret += res["ret"]
            stats.update(res["stats"])
            outcomes += res["outcomes"]
        return storage, adv, ret, stats, outcomes

    def close(self):
        for conn in self.conns:
            try:
                conn.send({"cmd": "stop"})
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


def make_worker_config(state_dim, action_dim, hidden, hist_hidden, seed, use_replay):
    return {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "history_dim": HISTORY_EVENT_DIM,
        "hidden": hidden,
        "hist_hidden": hist_hidden,
        "seed": seed,
        "shaping_beta": SHAPING_BETA,
        "win_bonus": WIN_BONUS,
        "use_lookahead": USE_LOOKAHEAD,
        "use_endgame_solver": USE_ENDGAME_SOLVER,
        "use_replay": use_replay,
    }


# ======================
# Resumable training state
# ======================

_STOP_REQUESTED = False


def _request_stop(signum, frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"\n[signal {signum}] checkpoint-and-exit requested; will save state after this iteration.", flush=True)


def save_training_state(path, model, optimizer, pool, replay_buffer, perf_history,
                        episode, best_eval_score, total_decisions, state_dim, action_dim):
    """Atomically persist everything needed to resume an interrupted run."""
    snapshots = [
        {"id": e.id, "wins": e.wins, "games": e.games, "state_dict": cpu_state_dict(e.model)}
        for e in pool._snapshot_entries()
    ]
    persistent = {eid: (e.wins, e.games) for eid, e in pool.entries.items() if e.kind != "snapshot"}
    blob = {
        "arch": "history_belief_ppo_v3",
        "model_state_dict": cpu_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "state_dim": state_dim, "action_dim": action_dim, "history_dim": HISTORY_EVENT_DIM,
        "hidden": getattr(model, "hidden", 256), "hist_hidden": getattr(model, "hist_hidden", 160),
        "episode": episode, "best_eval_score": best_eval_score, "total_decisions": total_decisions,
        "snap_counter": pool._snap_counter, "league_version": pool.version,
        "snapshots": snapshots,
        "baseline_state_dict": cpu_state_dict(pool.baseline) if pool.baseline is not None else None,
        "persistent_stats": persistent,
        "perf_history": perf_history,
        "replay_buffer": list(replay_buffer) if replay_buffer is not None else [],
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
    }
    tmp = path + ".tmp"
    torch.save(blob, tmp)
    os.replace(tmp, path)  # atomic: a SIGTERM mid-write can't corrupt the resume file


def _build_net(ckpt, sd, ad, hd):
    return HistoryBeliefPVNet(sd, ad, hd, hidden=ckpt["hidden"], hist_hidden=ckpt["hist_hidden"]).to(DEVICE)


def load_training_state(path, lr):
    # Our own trusted file; it holds RNG state + replay snapshots (Card objects),
    # so the torch>=2.6 weights_only default must be disabled.
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    sd, ad, hd = ckpt["state_dim"], ckpt["action_dim"], ckpt["history_dim"]
    model = _build_net(ckpt, sd, ad, hd)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    pool = LeaguePool()
    if ckpt.get("baseline_state_dict") is not None:
        bl = _build_net(ckpt, sd, ad, hd)
        bl.load_state_dict(ckpt["baseline_state_dict"])
        bl.eval()
        pool.baseline = bl
        pool.entries["baseline"] = OpponentEntry("baseline", "baseline", model=bl)
    pool._snap_counter = ckpt["snap_counter"]
    pool.version = ckpt["league_version"]
    for snap in ckpt["snapshots"]:
        m = _build_net(ckpt, sd, ad, hd)
        m.load_state_dict(snap["state_dict"])
        m.eval()
        pool.entries[snap["id"]] = OpponentEntry(snap["id"], "snapshot", model=m, wins=snap["wins"], games=snap["games"])
        pool.all.append(m)
        pool.recent.append(m)
        pool.best = m
    for eid, (w, g) in ckpt["persistent_stats"].items():
        if eid in pool.entries:
            pool.entries[eid].wins = w
            pool.entries[eid].games = g

    random.setstate(ckpt["rng_python"])
    np.random.set_state(ckpt["rng_numpy"])
    torch.set_rng_state(ckpt["rng_torch"].to("cpu") if hasattr(ckpt["rng_torch"], "to") else ckpt["rng_torch"])

    meta = {
        "episode": ckpt["episode"],
        "best_eval_score": ckpt["best_eval_score"],
        "total_decisions": ckpt["total_decisions"],
        "perf_history": ckpt["perf_history"],
        "replay_buffer": ckpt["replay_buffer"],
        "state_dim": sd, "action_dim": ad,
    }
    return model, optimizer, pool, meta


# ======================
# Main
# ======================


def initialize_model(state_dim: int, action_dim: int, init_checkpoint: Optional[str] = None, hidden: int = 256, hist_hidden: int = 160):
    if init_checkpoint is None:
        return HistoryBeliefPVNet(state_dim, action_dim, HISTORY_EVENT_DIM, hidden=hidden, hist_hidden=hist_hidden).to(DEVICE), None
    if not os.path.exists(init_checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {init_checkpoint}")
    model, ckpt = load_checkpoint(init_checkpoint)
    return model, {"path": init_checkpoint, "episode": ckpt.get("episode", "unknown")}


def _log_decision(infoset, action, human_seat):
    """Serialize one decision for the human-game dataset.

    Stores BOTH human-readable strings (for debugging / inspection) AND the pre-computed
    encoded feature arrays (state, history, per-action features, chosen-action index) so
    human_model.py can load and train directly without reconstructing infosets.
    """
    def cs(cards):
        return [str(c) for c in cards]
    lm = infoset.get("last_move")
    legal = infoset["legal_actions"]

    # --- encoded features (the trainable payload) ---
    state_vec = encode_state(infoset).tolist()
    hist_arr = encode_history_events(infoset).tolist()          # shape (T, 42) -- may be empty
    action_feats = [encode_move(a, infoset).tolist() for a in legal]
    # Chosen-action index into legal_actions list (-1 = pass, which maps to the last None entry)
    chosen_idx = -1
    for i, a in enumerate(legal):
        if a is None and action is None:
            chosen_idx = i; break
        if a is not None and action is not None:
            try:
                if (sorted((c.rank, c.suit or "") for c in a) ==
                        sorted((c.rank, c.suit or "") for c in action)):
                    chosen_idx = i; break
            except Exception:
                pass

    return {
        # --- human-readable (for inspection) ---
        "actor": infoset["player_index"],
        "is_human": int(infoset["player_index"] == human_seat),
        "hand": cs(infoset["hand"]),
        "hand_type": infoset.get("hand_type"),
        "last_move": cs(lm.cards) if lm is not None else None,
        "has_control": int(infoset.get("has_control", 0)),
        "current_pot": infoset.get("current_pot", 0),
        "deck_size": infoset.get("deck_size", 0),
        "points": infoset.get("points"),
        "opp_card_count": infoset.get("opp_card_count", 0),
        "played_cards": cs(infoset.get("played_cards", [])),
        "legal_actions_str": [None if a is None else cs(a) for a in legal],
        "action_str": None if action is None else cs(action),
        # --- trainable features ---
        "state": state_vec,
        "history": hist_arr,
        "action_feats": action_feats,   # list of 63-dim vectors, one per legal action
        "chosen_idx": chosen_idx,       # index into action_feats / legal_actions
        "n_legal": len(legal),
    }


def playtest(checkpoint_path: str = CHECKPOINT_BEST, bot_first: bool = False, search: bool = False,
             search_k: int = 16, log_dir: str = "runs/human_games"):
    model, ckpt = load_checkpoint(checkpoint_path)
    human = HumanPlayer()
    if search:
        # IS-MCTS: correct imperfect-info tree search, not PIMC
        base_bot = ISMCTSPlayer(model, device=DEVICE, iterations=max(60, search_k * 10))
    else:
        base_bot = ModelPlayer(model, device=DEVICE, training=False)
    bot = EndgameSolverPlayer(base_bot)  # exact, optimal play once the deck is empty
    bot.wants_concrete_same_rank_choices = getattr(base_bot, "wants_concrete_same_rank_choices", False)
    human_seat = 1 if bot_first else 0
    players = [bot, human] if bot_first else [human, bot]
    env = GameEnv(players, verbose=True)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Episode saved: {ckpt.get('episode', 'unknown')}")
    print(f"Bot mode: {'IS-MCTS (EXPERIMENTAL - underperforms policy)' if search else 'policy + exact endgame solver (default)'}")

    decisions = []
    step = 0
    while not env.done:
        print(f"\n=== ACTION {step} ===")
        print("Points:", env.points)
        print("Deck size:", env.deck.size())
        print("Current player:", env.current_player)
        infoset = env.get_infoset(env.current_player)
        act = players[env.current_player].act(infoset)
        decisions.append(_log_decision(infoset, act, human_seat))
        env.apply_action(act)
        step += 1

    print("\n=== GAME OVER ===")
    print("Final points:", env.points)

    os.makedirs(log_dir, exist_ok=True)
    rec = {
        "checkpoint": checkpoint_path, "human_seat": human_seat,
        "bot_mode": ("search" if search else "policy") + "+endgame",
        "final_points": {str(k): v for k, v in env.points.items()},
        "human_won": int(env.points[human_seat] - env.points[1 - human_seat] > 0),
        "decisions": decisions,
    }
    path = os.path.join(log_dir, f"game_{int(time.time())}.json")
    with open(path, "w") as f:
        json.dump(rec, f)
    print(f"Logged this game -> {path}  (human {'won' if rec['human_won'] else 'lost'})")


def main(device="auto", init_checkpoint=None, baseline_checkpoint=None, episodes=1000,
         rollouts_per_batch=ROLLOUTS_PER_BATCH, num_workers=0, lr=LR, epochs=EPOCHS,
         shaping_beta=SHAPING_BETA, win_bonus=WIN_BONUS, use_lookahead=USE_LOOKAHEAD,
         resume=False, state_freq=50, seed=0, out_dir=".", hidden=256, hist_hidden=160,
         use_endgame_solver=USE_ENDGAME_SOLVER):
    global DEVICE, SHAPING_BETA, WIN_BONUS, USE_LOOKAHEAD, LR, EPOCHS, USE_ENDGAME_SOLVER
    DEVICE = select_device(device)
    SHAPING_BETA, WIN_BONUS, USE_LOOKAHEAD, LR, EPOCHS = shaping_beta, win_bonus, use_lookahead, lr, epochs
    USE_ENDGAME_SOLVER = use_endgame_solver
    os.makedirs(out_dir, exist_ok=True)
    ckpt_latest = os.path.join(out_dir, CHECKPOINT_LATEST)
    ckpt_best = os.path.join(out_dir, CHECKPOINT_BEST)
    perf_csv = os.path.join(out_dir, PERF_CSV_PATH)
    perf_png = os.path.join(out_dir, PERF_PNG_PATH)
    state_path = os.path.join(out_dir, "train_state.pt")
    print(f"Using device: {DEVICE} | workers: {num_workers} | rollouts/batch: {rollouts_per_batch} "
          f"| lr: {lr} | epochs: {epochs} | shaping_beta: {shaping_beta} | win_bonus: {win_bonus} "
          f"| lookahead: {use_lookahead}", flush=True)

    signal.signal(signal.SIGTERM, _request_stop)
    try:
        signal.signal(signal.SIGUSR1, _request_stop)  # SLURM --signal target
    except (AttributeError, ValueError):
        pass

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dummy_env = GameEnv([RandomPlayer(), RandomPlayer()], verbose=False)
    dummy_infoset = dummy_env.get_infoset(dummy_env.current_player)
    state_dim = len(encode_state(dummy_infoset))
    action_dim = move_feature_dim()

    total_decisions = 0
    perf_history = make_perf_history()
    best_eval_score = float("-inf")
    start_ep = 0
    replay_buffer: Deque[dict] = deque(maxlen=REPLAY_BUFFER_SIZE)

    if resume and os.path.exists(state_path):
        model, optimizer, pool, meta = load_training_state(state_path, lr)
        start_ep = meta["episode"] + 1
        best_eval_score = meta["best_eval_score"]
        total_decisions = meta["total_decisions"]
        perf_history = meta["perf_history"]
        replay_buffer = deque(meta["replay_buffer"], maxlen=REPLAY_BUFFER_SIZE)
        state_dim, action_dim = meta["state_dim"], meta["action_dim"]
        print(f"Resumed from {state_path} at episode {start_ep} (best_score={best_eval_score:.3f}).", flush=True)
    else:
        model, init_meta = initialize_model(state_dim, action_dim, init_checkpoint, hidden=hidden, hist_hidden=hist_hidden)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        pool = LeaguePool()
        resolved_baseline = find_default_baseline(baseline_checkpoint)
        baseline_model, baseline_meta = maybe_load_baseline(resolved_baseline)
        pool.set_baseline(baseline_model if baseline_model is not None else model)
        if init_meta is not None:
            print(f"Initialized from {init_meta['path']} | episode={init_meta['episode']}", flush=True)
        if baseline_meta is not None:
            print(f"Using external baseline from {baseline_meta['path']} | episode={baseline_meta['episode']}", flush=True)
        else:
            print("No external baseline; using frozen initialization as baseline.", flush=True)

    # Adaptive-LR controller starts from the requested lr (overrides any lr restored
    # from a checkpoint's optimizer state, so the schedule is driven by KL from here).
    current_lr = lr
    for g in optimizer.param_groups:
        g["lr"] = current_lr

    collector = None
    if num_workers > 0:
        cfg = make_worker_config(state_dim, action_dim, getattr(model, "hidden", 256),
                                 getattr(model, "hist_hidden", 160), seed, use_replay=True)
        collector = ParallelCollector(num_workers, cfg)
        print(f"Spawned {num_workers} actor workers (spawn).", flush=True)

    exploiter = ExploiterAgent(state_dim, action_dim,
                               hidden=getattr(model, "hidden", 256),
                               hist_hidden=getattr(model, "hist_hidden", 160)) if EXPLOITER_ENABLED else None

    def _persist_state(ep):
        save_training_state(state_path, model, optimizer, pool, replay_buffer, perf_history,
                            ep, best_eval_score, total_decisions, state_dim, action_dim)

    try:
        for ep in range(start_ep, episodes):
            if _STOP_REQUESTED:
                print(f"Stopping at episode {ep}; saving state.", flush=True)
                _persist_state(ep - 1)
                break
            entropy_coef = max(ENTROPY_END, ENTROPY_START * (0.9985 ** ep))

            # ---- Exploiter training cadence ----
            if exploiter is not None and ep > 0 and ep % EXPLOITER_TRAIN_FREQ == 0:
                exploiter.update_target(model)
                exploiter.train(EXPLOITER_TRAIN_STEPS)
                exp_wr = exploiter.win_rate_vs_target()
                print(f"  [exploiter] ep={ep} WR vs main: {exp_wr:.2f}", flush=True)
                if exp_wr >= EXPLOITER_ADD_THRESHOLD:
                    pool.add_exploiter(exploiter.model)

            if ep > 0 and ep % SNAPSHOT_EVAL_FREQ == 0:
                if len(pool.all) < 3 or should_add_snapshot(model, pool):
                    pool.add(model)
                    added_snapshot = True
                else:
                    added_snapshot = False
            else:
                added_snapshot = False

            if collector is not None:
                batch_storage, batch_adv, batch_ret, stats, outcomes = collector.collect(model, pool, ep, rollouts_per_batch)
                for oc in outcomes:
                    pool.record_outcome(*oc)
            else:
                batch_storage, batch_adv, batch_ret = [], [], []
                stats = Counter()
                for _ in range(rollouts_per_batch):
                    s, adv, ret, rollout_stats, outcome = collect_rollout(model, pool, ep, replay_buffer=replay_buffer)
                    batch_storage += s
                    batch_adv += adv
                    batch_ret += ret
                    stats.update(rollout_stats)
                    pool.record_outcome(*outcome)

            states = [d.state for d in batch_storage]
            histories = [d.history for d in batch_storage]
            action_feats = [d.legal_action_feats for d in batch_storage]
            actions = [d.action for d in batch_storage]
            logp = [d.log_prob for d in batch_storage]
            aux_labels = [d.aux_labels for d in batch_storage]
            q_values = [d.q_values for d in batch_storage]
            old_values = [d.value for d in batch_storage]
            approx_kl = train_step(model, optimizer, (states, histories, action_feats, actions, logp, batch_ret, batch_adv, aux_labels, q_values, old_values), entropy_coef)
            if ADAPTIVE_LR:
                if approx_kl > 2 * TARGET_KL:
                    current_lr = max(LR_MIN, current_lr / 1.5)
                elif approx_kl < 0.5 * TARGET_KL:
                    current_lr = min(LR_MAX, current_lr * 1.05)
                for g in optimizer.param_groups:
                    g["lr"] = current_lr

            total_decisions += stats["decisions"]
            pass_rate = stats["passes"] / max(1, stats["decisions"])
            bomb_rate = stats["bombs"] / max(1, stats["decisions"])
            bomb_avail_use = stats["bombs"] / max(1, stats["bomb_opportunities"])

            if ep % 10 == 0:
                avg_ret = float(np.mean(batch_ret)) if batch_ret else 0.0
                print(
                    f"Episode {ep} | Avg Return: {avg_ret:.3f} | Decisions: {total_decisions} "
                    f"| Pass: {pass_rate:.2f} | Bomb: {bomb_rate:.2f} | BombAvailUse: {bomb_avail_use:.2f} "
                    f"| KL: {approx_kl:.4f} | LR: {current_lr:.2e} | Lookahead: {stats['lookahead_targets']}",
                    flush=True,
                )
            if ep % EVAL_PRINT_FREQ == 0:
                save_checkpoint(model, state_dim, action_dim, ep, ckpt_latest)
                wr_r = evaluate_random(model)
                eval_stats = refresh_pool_stats(model, pool)
                record_perf_point(perf_history, ep, wr_r, eval_stats)
                score = composite_eval_score(wr_r, eval_stats)
                if score > best_eval_score:
                    save_checkpoint(model, state_dim, action_dim, ep, ckpt_best)
                    best_eval_score = score
                mix = {k: stats[k] for k in stats if stats[k] > 0 and k not in {"decisions", "passes", "bombs", "bomb_opportunities", "lookahead_targets", "replay"} and not k.startswith("seat_")}
                print(
                    f"Eval | Random WR: {wr_r:.2f} | Baseline WR: {eval_stats['baseline_wr']:.2f} "
                    f"| Pool WR: {eval_stats['pool_wr']:.2f} | Best WR: {eval_stats['best_wr']:.2f} "
                    f"| Frontier WR: {eval_stats['frontier_wr']:.2f} | Scripted WR: {eval_stats['scripted_wr']:.2f} "
                    f"| League: {len(pool.all)} | BestScore: {best_eval_score:.3f} | AddedSnap: {added_snapshot}",
                    flush=True,
                )
                print(f"Opponent mix: {mix}", flush=True)
            if state_freq > 0 and ep > start_ep and ep % state_freq == 0:
                _persist_state(ep)
    finally:
        if collector is not None:
            collector.close()

    last_ep = min(episodes - 1, locals().get("ep", episodes - 1))
    save_checkpoint(model, state_dim, action_dim, last_ep, ckpt_latest)
    _persist_state(last_ep)
    if not perf_history["episode"] or perf_history["episode"][-1] != last_ep:
        record_perf_point(perf_history, last_ep, evaluate_random(model), refresh_pool_stats(model, pool))
    save_perf_csv(perf_history, perf_csv)
    save_perf_plot(perf_history, perf_png)
    print(f"Saved performance history -> {perf_csv} and plot -> {perf_png}", flush=True)


def evaluate_search(checkpoint_path, games=20, search_k=12):
    """Quantify the inference-time search benefit: raw policy vs PIMC search, both
    against the scripted archetypes."""
    model, _ = load_checkpoint(checkpoint_path)
    scripted = [OpponentSpec(name, make_scripted_builder(name)) for name in SCRIPTED_BUILDERS]
    raw = evaluate_opponents(model, scripted, games)

    def search_match(spec):
        wins = 0
        for _ in range(games):
            seat = random.randint(0, 1)
            players = [None, None]
            players[seat] = SearchModelPlayer(model, device=DEVICE, determinizations=search_k)
            players[1 - seat] = spec.make_player(model, DEVICE)
            env = GameEnv(players, verbose=False)
            while not env.done:
                env.apply_action(players[env.current_player].act(env.get_infoset(env.current_player)))
            wins += int(env.points[seat] - env.points[1 - seat] > 0)
        return wins / games

    search_wr = float(np.mean([search_match(s) for s in scripted]))
    print(f"Checkpoint {checkpoint_path}")
    print(f"  Raw policy   vs scripted: {raw:.3f}")
    print(f"  PIMC search  vs scripted: {search_wr:.3f}  (K={search_k})")
    print(f"  Search lift: {search_wr - raw:+.3f}")


# ======================
# CLI
# ======================


def parse_train_args(argv: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("init_checkpoint", nargs="?", default=None)
    parser.add_argument("--baseline-checkpoint", default=None)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--rollouts", type=int, default=ROLLOUTS_PER_BATCH, help="games collected per PPO update (total across workers)")
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) - 1 if os.environ.get("SLURM_CPUS_PER_TASK") else 0, help="parallel actor processes (0 = single-process)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--shaping-beta", type=float, default=SHAPING_BETA)
    parser.add_argument("--win-bonus", type=float, default=WIN_BONUS)
    parser.add_argument("--lookahead", action="store_true", help="enable (expensive) PIMC lookahead distillation during training")
    parser.add_argument("--no-endgame-solver", dest="endgame_solver", action="store_false", help="disable exact endgame-value bootstrapping")
    parser.set_defaults(endgame_solver=True)
    parser.add_argument("--resume", action="store_true", help="resume from <out-dir>/train_state.pt if present")
    parser.add_argument("--state-freq", type=int, default=50, help="episodes between training-state checkpoints (0 disables)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--hidden", type=int, default=256, help="MLP width (scale up on the cluster)")
    parser.add_argument("--hist-hidden", type=int, default=160, help="history GRU width")
    return parser.parse_args(argv)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "playtest":
        checkpoint = sys.argv[2] if len(sys.argv) >= 3 else CHECKPOINT_BEST
        rest = sys.argv[3:]
        bot_first = "bot_first" in rest
        # Default bot = policy + exact endgame solver (strong & fast). IS-MCTS is an
        # opt-in experiment ("mcts") that currently UNDERPERFORMS the policy.
        search = ("mcts" in rest) or ("search" in rest)
        playtest(checkpoint, bot_first=bot_first, search=search)
    elif len(sys.argv) >= 2 and sys.argv[1] == "evalsearch":
        checkpoint = sys.argv[2] if len(sys.argv) >= 3 else CHECKPOINT_BEST
        DEVICE = select_device("auto")
        evaluate_search(checkpoint)
    else:
        args = parse_train_args(sys.argv[1:])
        main(
            device=args.device,
            init_checkpoint=args.init_checkpoint,
            baseline_checkpoint=args.baseline_checkpoint,
            episodes=args.episodes,
            rollouts_per_batch=args.rollouts,
            num_workers=max(0, args.num_workers),
            lr=args.lr,
            epochs=args.epochs,
            shaping_beta=args.shaping_beta,
            win_bonus=args.win_bonus,
            use_lookahead=args.lookahead,
            resume=args.resume,
            state_freq=args.state_freq,
            seed=args.seed,
            out_dir=args.out_dir,
            hidden=args.hidden,
            hist_hidden=args.hist_hidden,
            use_endgame_solver=args.endgame_solver,
        )
