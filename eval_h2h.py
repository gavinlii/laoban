"""
Controlled head-to-head evaluation between two checkpoints, each played in its
OWN belief mode (fat = belief fusion on, lean = bypassed). Because USE_BELIEF is
a module global, we set it per-decision to the acting model's mode, so a single
game can fairly pit a belief-on model against a belief-off one.

Usage:
  python3 eval_h2h.py A.pt B.pt [--a-belief 1] [--b-belief 1] [-n 200] [--seed 0]

Reports A's win-rate and average point margin vs B (seats alternated).
"""
import argparse
import numpy as np
import torch

import train_ppo as T
from game import GameEnv
from encoder import encode_state, encode_move
from model import encode_history_events, HISTORY_EVENT_DIM
from train_ppo import HistoryBeliefPVNet


class _Dummy:
    # abstract same-rank action space (matches how the models were trained)
    wants_concrete_same_rank_choices = False
    def act(self, infoset):
        return infoset["legal_actions"][0]


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict") or ck.get("state_dict")
    m = HistoryBeliefPVNet(ck["state_dim"], ck["action_dim"],
                           ck.get("history_dim", HISTORY_EVENT_DIM),
                           hidden=ck.get("hidden", 256), hist_hidden=ck.get("hist_hidden", 160))
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m, ck.get("episode", "?")


def greedy(model, info, belief):
    T.USE_BELIEF = belief
    legal = info["legal_actions"]
    st = torch.from_numpy(encode_state(info)).float().unsqueeze(0)
    h = encode_history_events(info)
    ht = torch.from_numpy(h if len(h) else np.zeros((0, HISTORY_EVENT_DIM), dtype=np.float32)).float().unsqueeze(0)
    hl = torch.tensor([len(h)], dtype=torch.long)
    af = np.stack([encode_move(a, info) for a in legal]).astype(np.float32)
    at = torch.from_numpy(af).float().unsqueeze(0)
    with torch.no_grad():
        logits, _, _ = model.score_actions(st, ht, hl, at)
    return legal[int(torch.argmax(logits[0]).item())]


def h2h(mA, bA, mB, bB, n=200, seed0=0):
    a_wins = b_wins = 0
    a_margin = 0.0
    for g in range(n):
        env = GameEnv([_Dummy(), _Dummy()], seed=seed0 + g, verbose=False, history_maxlen=200)
        a_seat = g % 2
        steps = 0
        while not env.done and steps < 600:
            p = env.current_player
            info = env.get_infoset(p)
            if p == a_seat:
                act = greedy(mA, info, bA)
            else:
                act = greedy(mB, info, bB)
            env.apply_action(act)
            steps += 1
        margin = env.points[a_seat] - env.points[1 - a_seat]
        a_margin += margin
        if margin > 0: a_wins += 1
        elif margin < 0: b_wins += 1
    return a_wins / n, a_margin / n, (a_wins, b_wins, n - a_wins - b_wins)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--a-belief", type=int, default=1)
    ap.add_argument("--b-belief", type=int, default=1)
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    mA, eA = load(args.a)
    mB, eB = load(args.b)
    wr, margin, (aw, bw, ties) = h2h(mA, bool(args.a_belief), mB, bool(args.b_belief), n=args.n, seed0=args.seed)
    se = (wr * (1 - wr) / args.n) ** 0.5
    print(f"A = {args.a} (ep {eA}, belief={bool(args.a_belief)})")
    print(f"B = {args.b} (ep {eB}, belief={bool(args.b_belief)})")
    print(f"games={args.n}  A wins={aw} B wins={bw} ties={ties}")
    print(f"A win-rate = {wr:.3f} +/- {se:.3f}   A avg margin = {margin:+.1f}")
