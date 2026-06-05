"""
policy_loader.py — loads a V5 HistoryBeliefPVNet checkpoint and exposes
choose_action(infoset) for use by the web backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import encoder as enc
from model import HistoryBeliefPVNet, HISTORY_EVENT_DIM, encode_history_events

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

try:
    torch.set_num_threads(min(4, torch.get_num_threads()))
except Exception:
    pass


@dataclass
class LoadedPolicy:
    path: str
    model: HistoryBeliefPVNet
    state_dim: int
    action_dim: int
    episode: object

    def choose_action(self, infoset) -> dict:
        legal = infoset["legal_actions"]
        state  = enc.encode_state(infoset)
        history = encode_history_events(infoset)
        action_feats = np.stack([enc.encode_move(a, infoset) for a in legal]).astype(np.float32)

        with torch.no_grad():
            st = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
            hist_arr = history if len(history) else np.zeros((0, HISTORY_EVENT_DIM), dtype=np.float32)
            ht  = torch.from_numpy(hist_arr).float().unsqueeze(0).to(DEVICE)
            hl  = torch.tensor([len(history)], dtype=torch.long, device=DEVICE)
            at  = torch.from_numpy(action_feats).float().unsqueeze(0).to(DEVICE)
            logits, values, _ = self.model.score_actions(st, ht, hl, at)
            idx = int(torch.argmax(logits[0]).item())
            value = float(values[0].item())

        return {
            "index": idx,
            "move": legal[idx],
            "value": value,
        }


def load_policy(checkpoint_path: str) -> LoadedPolicy:
    path = str(Path(checkpoint_path).expanduser().resolve())
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)

    state_dim  = int(ckpt["state_dim"])
    action_dim = int(ckpt["action_dim"])

    # Validate dims match the current encoder
    from game import GameEnv, RandomPlayer
    _env = GameEnv([RandomPlayer(), RandomPlayer()], verbose=False)
    _info = _env.get_infoset(_env.current_player)
    expected_state  = len(enc.encode_state(_info))
    expected_action = enc.move_feature_dim()
    if state_dim != expected_state or action_dim != expected_action:
        raise ValueError(
            f"Checkpoint dims ({state_dim}/{action_dim}) don't match "
            f"current encoder ({expected_state}/{expected_action}). "
            f"Ensure policy.pt was trained with this encoder."
        )

    sd = ckpt.get("model_state_dict") or ckpt.get("state_dict")
    if sd is None:
        raise ValueError("Checkpoint has no 'model_state_dict' or 'state_dict' key.")

    hidden      = int(sd["state_net.0.weight"].shape[0])
    hist_hidden = int(sd["history_rnn.weight_ih_l0"].shape[1])

    model = HistoryBeliefPVNet(state_dim, action_dim, HISTORY_EVENT_DIM,
                               hidden=hidden, hist_hidden=hist_hidden).to(DEVICE)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint load mismatch — missing: {missing}, unexpected: {unexpected}")
    model.eval()

    return LoadedPolicy(
        path=path,
        model=model,
        state_dim=state_dim,
        action_dim=action_dim,
        episode=ckpt.get("episode", "unknown"),
    )
