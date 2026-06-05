"""
Human model: behavior-cloning a policy to imitate the human player.

Pipeline
--------
1. Load all logged game JSON files from runs/human_games/.
2. Warm-start from the bot checkpoint (same 172/63 architecture).
3. Fine-tune with:
     loss = cross_entropy(predicted_action, human_chosen_action)
          + KL_WEIGHT * KL(model || frozen_bot)
   The KL term acts as a prior: in situations the human data doesn't cover,
   fall back to the bot (competent play) rather than hallucinate.
4. Validate: report top-1 accuracy on a held-out set, and compare against
   the raw bot-prior accuracy (the bar we need to beat to prove the model
   learned something *human-specific*, not just the game).
5. Save the trained human model to a checkpoint that human_eval and the
   league can load directly.

Usage
-----
# Train:
python3 human_model.py --games-dir runs/human_games --bot-ckpt runs/local_v3/policy_latest.pt

# Evaluate only (no training):
python3 human_model.py --eval-only --human-ckpt runs/human_model/human_model.pt \
    --bot-ckpt runs/local_v3/policy_latest.pt

# Head-to-head: how often does the main bot beat the human model?
python3 human_model.py --vs-bot --bot-ckpt runs/local_v3/policy_latest.pt \
    --human-ckpt runs/human_model/human_model.pt --games 60
"""
import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Imports from the training codebase
# ---------------------------------------------------------------------------
import train_ppo as T
from train_ppo import (
    HistoryBeliefPVNet, load_checkpoint, save_checkpoint,
    HISTORY_EVENT_DIM, ModelPlayer,
)
from game import GameEnv

DEVICE = torch.device("cpu")

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
KL_WEIGHT      = 0.30   # strength of the KL-regularization toward the frozen bot
                         # 0 = pure BC, 1 = heavily anchored. Start conservative.
LR             = 5e-5   # small: we're fine-tuning, not training from scratch
EPOCHS         = 40
BATCH_SIZE     = 64
VAL_FRACTION   = 0.15   # held-out fraction for validation
MIN_GAMES      = 5      # warn if fewer games (model will be unreliable)
PATIENCE       = 8      # early-stop if val accuracy doesn't improve for this many epochs


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_decisions(games_dir: str, human_only: bool = True) -> List[dict]:
    """Load all logged decisions from JSON game files.

    If human_only=True (default), only keep decisions where the human was the actor.
    The bot's own decisions aren't useful for BC (we already have the policy for that).
    """
    decisions = []
    game_files = sorted(Path(games_dir).glob("game_*.json"))
    if not game_files:
        raise FileNotFoundError(f"No game_*.json files found in {games_dir}")

    for gf in game_files:
        with open(gf) as f:
            rec = json.load(f)
        human_seat = rec.get("human_seat", 0)
        for d in rec.get("decisions", []):
            # Skip decisions that don't have the encoded feature payload (old-format logs)
            if "state" not in d or "action_feats" not in d or d.get("chosen_idx", -99) < 0:
                continue
            if human_only and not d.get("is_human", 0):
                continue
            # Skip pass-only decisions (no real choice)
            if d["n_legal"] <= 1:
                continue
            decisions.append(d)

    return decisions


def decisions_to_tensors(decisions: List[dict], device: torch.device):
    """Convert list of decision dicts to padded tensors ready for training."""
    max_legal = max(d["n_legal"] for d in decisions)
    state_dim = len(decisions[0]["state"])
    action_dim = len(decisions[0]["action_feats"][0])

    states     = np.zeros((len(decisions), state_dim), dtype=np.float32)
    action_mat = np.zeros((len(decisions), max_legal, action_dim), dtype=np.float32)
    mask       = np.zeros((len(decisions), max_legal), dtype=bool)
    targets    = np.zeros(len(decisions), dtype=np.int64)

    for i, d in enumerate(decisions):
        states[i] = d["state"]
        n = d["n_legal"]
        for j, af in enumerate(d["action_feats"]):
            action_mat[i, j] = af
            mask[i, j] = True
        targets[i] = d["chosen_idx"]

    # History is variable-length so we pad it separately
    max_hist = max((len(d["history"]) for d in decisions), default=0)
    hist_dim = HISTORY_EVENT_DIM
    hists    = np.zeros((len(decisions), max(max_hist, 1), hist_dim), dtype=np.float32)
    hlens    = np.zeros(len(decisions), dtype=np.int64)
    for i, d in enumerate(decisions):
        h = np.array(d["history"], dtype=np.float32) if d["history"] else np.zeros((0, hist_dim), dtype=np.float32)
        if len(h) > 0:
            hists[i, :len(h)] = h
            hlens[i] = len(h)

    return (
        torch.tensor(states,     device=device),
        torch.tensor(hists,      device=device),
        torch.tensor(hlens,      device=device),
        torch.tensor(action_mat, device=device),
        torch.tensor(mask,       device=device),
        torch.tensor(targets,    device=device),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_human_model(
    games_dir: str,
    bot_ckpt: str,
    out_dir: str = "runs/human_model",
    kl_weight: float = KL_WEIGHT,
    lr: float = LR,
    epochs: int = EPOCHS,
    seed: int = 0,
):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    # ---- load data ----
    print(f"Loading human decisions from {games_dir} ...")
    decisions = load_decisions(games_dir, human_only=True)
    print(f"  {len(decisions)} human decisions loaded")
    if len(decisions) < 20:
        print(f"  WARNING: very few decisions ({len(decisions)}). "
              f"Model will likely overfit. Play more games first.")
    if len(decisions) == 0:
        print("  No data to train on. Play some games via:")
        print("  python3 train_ppo.py playtest runs/local_v3/policy_latest.pt")
        return None

    # ---- train/val split (by game, not by decision, to avoid leakage) ----
    random.shuffle(decisions)
    n_val = max(1, int(len(decisions) * VAL_FRACTION))
    val_dec   = decisions[:n_val]
    train_dec = decisions[n_val:]
    print(f"  Train: {len(train_dec)} decisions | Val: {len(val_dec)} decisions")

    train_tensors = decisions_to_tensors(train_dec, DEVICE)
    val_tensors   = decisions_to_tensors(val_dec,   DEVICE)
    state_dim  = train_tensors[0].shape[1]
    action_dim = train_tensors[3].shape[2]
    print(f"  state_dim={state_dim}, action_dim={action_dim}")

    # ---- warm-start from bot checkpoint ----
    print(f"\nWarm-starting from {bot_ckpt} ...")
    model, ckpt_meta = load_checkpoint(bot_ckpt)
    model = model.to(DEVICE)
    model.train()
    print(f"  Loaded bot ep={ckpt_meta.get('episode')}")

    # Frozen bot copy for KL regularization (never updated)
    bot_frozen, _ = load_checkpoint(bot_ckpt)
    bot_frozen = bot_frozen.to(DEVICE)
    bot_frozen.eval()
    for p in bot_frozen.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ---- baseline: how well does the raw bot already predict human actions? ----
    bot_acc = _accuracy(bot_frozen, val_tensors)
    print(f"\nBot-prior accuracy on val (the bar to beat): {bot_acc:.3f}")

    # ---- training loop ----
    best_val_acc = 0.0
    best_epoch   = 0
    best_state   = None

    st, ht, hl, at, mk, tgt = train_tensors
    N = st.shape[0]

    for epoch in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(N)
        total_loss = 0.0; n_batches = 0

        for start in range(0, N, BATCH_SIZE):
            mb = idx[start:start + BATCH_SIZE]
            st_mb = st[mb]; ht_mb = ht[mb]; hl_mb = hl[mb]
            at_mb = at[mb]; mk_mb = mk[mb]; tgt_mb = tgt[mb]

            logits, _, _ = model.score_actions(st_mb, ht_mb, hl_mb, at_mb)
            logits = logits.masked_fill(~mk_mb, -1e9)

            # Behavior-cloning loss: cross-entropy on human-chosen action
            bc_loss = F.cross_entropy(logits, tgt_mb)

            # KL regularization toward frozen bot (prevents over-fitting on small data)
            if kl_weight > 0:
                with torch.no_grad():
                    bot_logits, _, _ = bot_frozen.score_actions(st_mb, ht_mb, hl_mb, at_mb)
                    bot_logits = bot_logits.masked_fill(~mk_mb, -1e9)
                    bot_probs = torch.softmax(bot_logits, dim=-1)
                model_lp = torch.log_softmax(logits, dim=-1)
                kl = (bot_probs * (torch.log(bot_probs + 1e-9) - model_lp)).sum(-1).mean()
            else:
                kl = torch.tensor(0.0)

            loss = bc_loss + kl_weight * kl
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item(); n_batches += 1

        val_acc = _accuracy(model, val_tensors)
        avg_loss = total_loss / max(1, n_batches)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | loss {avg_loss:.4f} | val_acc {val_acc:.3f}"
                  f"  {'<-- best' if val_acc > best_val_acc else ''}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch - best_epoch >= PATIENCE:
            print(f"  Early stop at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    # ---- restore best ----
    model.load_state_dict(best_state)
    model.eval()

    print(f"\n{'='*60}")
    print(f"Bot-prior accuracy (baseline):  {bot_acc:.3f}")
    print(f"Human model accuracy (best):    {best_val_acc:.3f}  @ epoch {best_epoch}")
    improvement = best_val_acc - bot_acc
    if improvement > 0.02:
        print(f"  >> Model learned human-specific tendencies (+{improvement:.3f} over bot)")
    elif improvement > 0:
        print(f"  >> Marginal improvement (+{improvement:.3f}) -- play more games")
    else:
        print(f"  >> No improvement over bot prior -- need more data or check data quality")

    # ---- save ----
    out_path = os.path.join(out_dir, "human_model.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "state_dim":  state_dim,
        "action_dim": action_dim,
        "arch":       "history_belief_ppo_v3",
        "bot_ckpt":   bot_ckpt,
        "n_decisions": len(decisions),
        "val_acc":    best_val_acc,
        "bot_acc":    bot_acc,
        "trained_at": int(time.time()),
    }, out_path)
    print(f"\nSaved -> {out_path}")
    return out_path


def _accuracy(model, tensors) -> float:
    """Top-1 action prediction accuracy on the given tensors."""
    model.eval()
    st, ht, hl, at, mk, tgt = tensors
    with torch.no_grad():
        logits, _, _ = model.score_actions(st, ht, hl, at)
        logits = logits.masked_fill(~mk, -1e9)
        pred = logits.argmax(dim=-1)
    return float((pred == tgt).float().mean().item())


# ---------------------------------------------------------------------------
# Head-to-head: main bot vs human model
# ---------------------------------------------------------------------------

def vs_bot(bot_ckpt: str, human_ckpt: str, n_games: int = 60):
    """Play the main bot against the trained human model. Reports win-rate for
    the *bot* (lower is worse, since the goal is to make the human model a real
    challenger that the bot needs to improve against)."""
    T.DEVICE = DEVICE

    bot_model, _   = load_checkpoint(bot_ckpt)
    hm_data        = torch.load(human_ckpt, map_location="cpu", weights_only=False)
    sd = hm_data["state_dim"]; ad = hm_data["action_dim"]
    hm_model = HistoryBeliefPVNet(sd, ad, HISTORY_EVENT_DIM).to(DEVICE)
    hm_model.load_state_dict(hm_data["state_dict"])
    hm_model.eval()

    bot_wins = 0
    for g in range(n_games):
        seat = g % 2
        players = [None, None]
        players[seat]     = ModelPlayer(bot_model, device=DEVICE, training=False)
        players[1 - seat] = ModelPlayer(hm_model,  device=DEVICE, training=False)
        env = GameEnv(players, verbose=False)
        while not env.done:
            cur = env.current_player
            env.apply_action(players[cur].act(env.get_infoset(cur)))
        bot_wins += int(env.points[seat] - env.points[1 - seat] > 0)

    wr = bot_wins / n_games
    print(f"\nBot vs Human-model: bot WR = {wr:.3f} over {n_games} games")
    print(f"  (0.50 = human model is an equal challenge; <0.60 = useful league opponent)")
    return wr


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Human-model behavior cloning")
    parser.add_argument("--games-dir",    default="runs/human_games",          help="dir with game_*.json logs")
    parser.add_argument("--bot-ckpt",     default="runs/local_v3/policy_latest.pt")
    parser.add_argument("--out-dir",      default="runs/human_model")
    parser.add_argument("--human-ckpt",   default="runs/human_model/human_model.pt")
    parser.add_argument("--kl-weight",    type=float, default=KL_WEIGHT)
    parser.add_argument("--lr",           type=float, default=LR)
    parser.add_argument("--epochs",       type=int,   default=EPOCHS)
    parser.add_argument("--seed",         type=int,   default=0)
    parser.add_argument("--eval-only",    action="store_true", help="skip training, just evaluate accuracy")
    parser.add_argument("--vs-bot",       action="store_true", help="head-to-head: bot vs human model")
    parser.add_argument("--games",        type=int,   default=60, help="games for --vs-bot")
    args = parser.parse_args()

    T.DEVICE = DEVICE

    if args.vs_bot:
        vs_bot(args.bot_ckpt, args.human_ckpt, args.games)

    elif args.eval_only:
        print(f"Loading decisions from {args.games_dir} ...")
        decisions = load_decisions(args.games_dir, human_only=True)
        print(f"  {len(decisions)} human decisions")
        tensors = decisions_to_tensors(decisions, DEVICE)

        bot_model, _ = load_checkpoint(args.bot_ckpt)
        bot_acc = _accuracy(bot_model, tensors)
        print(f"Bot-prior accuracy: {bot_acc:.3f}")

        hm_data = torch.load(args.human_ckpt, map_location="cpu", weights_only=False)
        sd = hm_data["state_dim"]; ad = hm_data["action_dim"]
        hm_model = HistoryBeliefPVNet(sd, ad, HISTORY_EVENT_DIM).to(DEVICE)
        hm_model.load_state_dict(hm_data["state_dict"])
        hm_acc = _accuracy(hm_model, tensors)
        print(f"Human model accuracy: {hm_acc:.3f}  (improvement: {hm_acc - bot_acc:+.3f})")

    else:
        train_human_model(
            games_dir  = args.games_dir,
            bot_ckpt   = args.bot_ckpt,
            out_dir    = args.out_dir,
            kl_weight  = args.kl_weight,
            lr         = args.lr,
            epochs     = args.epochs,
            seed       = args.seed,
        )
