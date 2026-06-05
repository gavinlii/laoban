"""
Potential-based reward shaping (PBRS) for Five-Ten-K.

The raw reward is the change in point margin. That signal heavily under-credits
*card economy*: spending an Ace / 2 / joker to win a zero-point trick costs ~0
points immediately, but throws away the option value of a control card that
could have captured a big pot later. Under GAMMA=0.997 and small batches the
value net never learns this, which is exactly the reported failure mode
("plays high cards and points on absolutely nothing").

We add F(s,s') = GAMMA * Phi(s') - Phi(s), where Phi is a function of the
acting player's hand only. By the Ng-Harada-Russell (1999) theorem this leaves
the optimal policy unchanged while injecting a dense, local signal: spending a
control card drops Phi immediately, so unless the move earns enough points to
offset it, the shaped reward is negative.

Because hands refill from the deck between a player's decisions, Phi naturally
encodes "spending control is cheap while the deck is deep (you'll redraw) and
expensive in the endgame (no redraws left)" -- which is the correct economics.
PBRS invariance still holds: the per-decision shaping telescopes to -Phi(s_0)
in expectation regardless of why Phi changes.
"""
from collections import Counter

# Control (utility) value per rank. Only A and above carry control value; the
# numbers are in "card-utility units" roughly commensurate with points so the
# shaping coefficient is interpretable.
CONTROL_UTILITY = {
    14: 1.0,   # Ace
    17: 1.6,   # 2  (highest natural rank in this game)
    20: 2.2,   # small joker
    30: 3.2,   # big joker
}

# A held bomb is latent optionality worth more than its raw cards.
BOMB_BONUS = 1.5


def control_value(hand):
    return sum(CONTROL_UTILITY.get(c.rank, 0.0) for c in hand)


def bomb_count(hand):
    counts = Counter(c.rank for c in hand)
    bombs = sum(1 for v in counts.values() if v == 4)
    if counts.get(20, 0) and counts.get(30, 0):
        bombs += 1
    if counts.get(5, 0) and counts.get(10, 0) and counts.get(13, 0):
        bombs += 1
    return bombs


def hand_potential_raw(hand):
    """Latent value of the cards a player is holding, in card-utility units."""
    if not hand:
        return 0.0
    return control_value(hand) + BOMB_BONUS * bomb_count(hand)


def scaled_potential(hand, beta, point_scale):
    """Phi(s) in the same units as the (margin / point_scale) reward."""
    if beta <= 0.0:
        return 0.0
    return beta * hand_potential_raw(hand) / point_scale


def apply_shaping(rewards, phis, gamma):
    """
    In-place PBRS over a single player's decision sequence.

    rewards[t], phis[t] are aligned per model decision; phis[t] = Phi(s_t) on the
    hand *before* the move. The state after the last decision is terminal, so
    Phi(s_{T}) = 0.
    """
    n = len(rewards)
    for t in range(n):
        phi_t = phis[t]
        phi_next = phis[t + 1] if t + 1 < n else 0.0
        rewards[t] += gamma * phi_next - phi_t
    return rewards
