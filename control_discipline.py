"""
Direct measurement of control-card discipline -- the behavior the win-rate-vs-
deployed metric is structurally blind to (in self-play, wasting control isn't
punished). Plays a policy in self-play and, at every FOLLOWING decision where the
pot is cheap (<5 pts) AND the policy had an out (a legal pass or a non-control,
non-bomb alternative), measures how often it overspends a control card (rank>=A)
or a bomb anyway. Lower waste-rate = better discipline = the targeted improvement.

Usage: python3 control_discipline.py <ckpt> --belief {0,1} [-n 150]
"""
import argparse
import numpy as np
import torch

import train_ppo as T
from game import GameEnv, Move, POINT_VALUES
from eval_h2h import load, greedy, _Dummy

HIGH = 14  # Ace and above count as control (A=14, 2=17, jokers 20/30)


def _cards(m):
    return m.cards if isinstance(m, Move) else m


def is_control(m):
    return m is not None and any(c.rank >= HIGH for c in _cards(m))


def is_joker(m):
    return m is not None and any(c.rank >= 20 for c in _cards(m))


def is_bomb(m):
    return m is not None and (m if isinstance(m, Move) else Move(m)).type == "bomb"


def probe(model, belief, n=150, seed0=0):
    temptations = wastes = joker_wastes = 0
    control_commits = 0
    pots_when_control = []
    for g in range(n):
        env = GameEnv([_Dummy(), _Dummy()], seed=seed0 + g, verbose=False, history_maxlen=200)
        steps = 0
        while not env.done and steps < 600:
            p = env.current_player
            info = env.get_infoset(p)
            legal = info["legal_actions"]
            move = greedy(model, info, belief)
            pot = env.current_pot
            leading = info.get("hand_type") is None
            can_pass = None in legal
            has_cheap = any((a is not None) and not is_control(a) and not is_bomb(a) for a in legal)
            if not leading:
                committed = is_control(move) or is_bomb(move)
                if committed:
                    control_commits += 1
                    pots_when_control.append(pot)
                # "temptation": cheap pot, and a cheaper way out existed
                if pot < 5 and (can_pass or has_cheap):
                    temptations += 1
                    if committed:
                        wastes += 1
                    if is_joker(move) and pot == 0:
                        joker_wastes += 1
            env.apply_action(move)
            steps += 1
    return {
        "temptations": temptations,
        "waste_rate": wastes / max(1, temptations),
        "joker_waste_per_game": joker_wastes / n,
        "avg_pot_when_committing_control": float(np.mean(pots_when_control)) if pots_when_control else 0.0,
        "control_commits": control_commits,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--belief", type=int, default=1)
    ap.add_argument("-n", type=int, default=150)
    args = ap.parse_args()
    m, ep = load(args.ckpt)
    s = probe(m, bool(args.belief), n=args.n)
    print(f"{args.ckpt} (ep {ep}, belief={bool(args.belief)}), {args.n} games:")
    print(f"  waste-rate (overspend control/bomb on cheap pot w/ an out): {s['waste_rate']:.3f}  over {s['temptations']} temptations")
    print(f"  joker-on-zero-pot per game: {s['joker_waste_per_game']:.3f}")
    print(f"  avg pot when committing a control card/bomb: {s['avg_pot_when_committing_control']:.1f}  (higher=more disciplined)")
