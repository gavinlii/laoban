"""
Behavioral diagnostics for the Five-Ten-K policy.

Reconstructs tricks from GameEnv.public_history and measures the specific
failure modes reported: wasting high/utility cards and feeding points.
Run: python3 diagnose_policy.py [checkpoint.pt] [--games N] [--opp random|self]
"""
import argparse
import random
from collections import defaultdict

import numpy as np
import torch

from game import GameEnv, RandomPlayer, POINT_VALUES
from train_ppo import ModelPlayer, load_checkpoint, DEVICE

HIGH = 14  # Ace and above (A=14, 2=17, jokers=20/30) are "control" cards


def reconstruct_tricks(history):
    """Yield dicts describing each trick: leader, plays [(actor,rank,pts,len,bomb)], winner, points."""
    tricks = []
    cur = None
    for ev in history:
        k = ev["kind"]
        if k == "play":
            actor = ev["actor"]
            if cur is None:
                cur = {"leader": actor, "plays": [], "points": 0}
            cur["plays"].append((actor, ev.get("move_rank", 0), ev.get("points_gained", 0),
                                 ev.get("move_len", 1), ev.get("is_bomb", 0)))
            cur["points"] += ev.get("points_gained", 0)
        elif k in ("pass",):
            if cur is not None and cur["plays"]:
                cur["winner"] = cur["plays"][-1][0]
                tricks.append(cur)
            cur = None
        elif k == "terminal":
            if cur is not None and cur["plays"]:
                cur["winner"] = ev.get("winner", cur["plays"][-1][0])
                tricks.append(cur)
            cur = None
        elif k in ("hand_end", "draw", "deal"):
            # hand_end already handled by the preceding pass; just reset
            cur = None
    return tricks


def play_game(model, opp_kind):
    model_seat = random.randint(0, 1)
    bot = ModelPlayer(model, device=DEVICE, training=False)
    if opp_kind == "self":
        opp = ModelPlayer(model, device=DEVICE, training=False)
    else:
        opp = RandomPlayer()
    players = [None, None]
    players[model_seat] = bot
    players[1 - model_seat] = opp
    env = GameEnv(players, verbose=False, history_maxlen=100000)
    while not env.done:
        info = env.get_infoset(env.current_player)
        act = players[env.current_player].act(info)
        env.apply_action(act)
    return env, model_seat


def analyze(checkpoint, games, opp_kind):
    model, ckpt = load_checkpoint(checkpoint)
    model.eval()
    agg = defaultdict(float)
    margins = []
    wins = 0
    for _ in range(games):
        env, seat = play_game(model, opp_kind)
        margin = env.points[seat] - env.points[1 - seat]
        margins.append(margin)
        wins += int(margin > 0)
        tricks = reconstruct_tricks(list(env.public_history))
        for t in tricks:
            leader = t["leader"]
            winner = t.get("winner", leader)
            for who, tag in [(seat, "model"), (1 - seat, "opp")]:
                # leads
                if leader == who:
                    agg[f"{tag}_leads"] += 1
                    lead_actor, lead_rank, _, lead_len, lead_bomb = t["plays"][0]
                    if not lead_bomb and lead_rank >= HIGH:
                        agg[f"{tag}_high_leads"] += 1
                # points fed (points this player played into a trick they lost)
                fed = sum(p[2] for p in t["plays"] if p[0] == who)
                if winner != who:
                    agg[f"{tag}_fed_points"] += fed
                else:
                    agg[f"{tag}_captured_points"] += t["points"]
                # high cards spent to win a zero-point trick
                if winner == who and t["points"] == 0:
                    spent_high = any((not p[4] and p[1] >= HIGH) for p in t["plays"] if p[0] == who)
                    if spent_high:
                        agg[f"{tag}_high_on_zero"] += 1
    g = games
    print(f"\n=== {checkpoint} vs {opp_kind} over {g} games (saved ep={ckpt.get('episode')}) ===")
    print(f"Win rate: {wins/g:.3f} | Avg margin: {np.mean(margins):+.2f} (±{np.std(margins):.1f})")
    for tag in ["model", "opp"]:
        leads = max(1.0, agg[f"{tag}_leads"])
        print(
            f"  [{tag:5s}] high-lead rate: {agg[f'{tag}_high_leads']/leads:.3f} "
            f"({agg[f'{tag}_high_leads']:.0f}/{leads:.0f}) | "
            f"high-card-wins-0pt/game: {agg[f'{tag}_high_on_zero']/g:.2f} | "
            f"pts fed/game: {agg[f'{tag}_fed_points']/g:.1f} | "
            f"pts captured/game: {agg[f'{tag}_captured_points']/g:.1f}"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", default="policy_latest.pt")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--opp", default="random", choices=["random", "self"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    analyze(args.checkpoint, args.games, args.opp)
