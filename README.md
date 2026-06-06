# Laoban · 老板

A playable game engine for the Chinese card game **5/10/K** (五十K). The bot is a PPO-trained policy with a GRU history encoder, opponent-hand belief heads, and an exact endgame minimax solver.

Live at **[laoban.cards](https://laoban.cards)**

---

## Migration note

This repo holds the **current (post-2026-05-05) architecture**. All earlier work — the legacy `ActionConditionedPVNet` backend, the pre-V5 encoders, and training history through V4 — lives in **`laoban_legacy`**. Development moved here when the policy was upgraded to the V5 `HistoryBeliefPVNet` architecture and the web backend was rebuilt to match it. If you're looking for anything dated before 2026-05-05, it's in the legacy repo.

---

## The policy

**`HistoryBeliefPVNet`** — `policy.pt`, episode 8251, dims 172/63, ~865K params.

| Component | Detail |
|---|---|
| State encoder | 172-dim hand/board features |
| Action encoder | 63-dim per-move features |
| History encoder | GRU over 42-dim public-event vectors (≤48 events) |
| Belief heads | Opponent rank-counts, bomb/empty/points estimates — supervised by privileged labels, detached, fused into the decision context |
| Policy / value heads | Score each legal action; value head bootstrapped with the exact endgame value |

**Endgame solver** (`endgame.py`): once the draw pile empties the game is perfect-information, so the opponent's hand is known exactly. The bot switches to a memoized minimax search (~24 ms for 5v5) for optimal run-out play. 

## Game rules (brief)

2 players, 54-card deck (two jokers). Point cards: **5 = 5**, **10 = 10**, **K = 10** (120 pts total).
- Plays: singles, pairs, triples, 5-card straights, and bombs.
- A reply must be the same type and higher — or a bomb. A single PASS ends the trick.
- Bombs (weakest→strongest): off-suit 5-10-K < suited 5-10-K < four-of-a-kind < both jokers.
- After each trick the winner refills first; both refill to 5 while the deck lasts.
- Endgame: once the deck is empty, the first to empty their hand gains **+20**; the loser's remaining point cards are deducted.
- Most points wins.
