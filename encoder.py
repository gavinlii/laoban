import numpy as np
from collections import Counter
from functools import lru_cache
from game import Card, Move, MoveGenerator, RANKS, POINT_VALUES

RANK_TO_IDX = {r: i for i, r in enumerate(RANKS)}

MOVE_TYPES = ["none", "single", "pair", "triple", "straight", "bomb"]
MOVE_TO_IDX = {m: i for i, m in enumerate(MOVE_TYPES)}

MOVE_CONTEXT_DIM = 9
HAND_SUMMARY_DIM = 13
STATE_ENDGAME_DIM = 10
ACTION_ENDGAME_DIM = 10
CONSEQUENCE_DIM = 6
ACTION_EXTRA_DIM = 4
ACTION_RISK_DIM = 4
BELIEF_PRIOR_DIM = 8
SUIT_510K_DIM = 4
TEMPO_DIM = 3
BOSS_DIM = 3
DECISIVE_DIM = 3

CONTROL_RANK = 14  # Ace and above are control cards
TOTAL_CONTROL_CARDS = 10.0  # 4 aces + 4 twos + 2 jokers
FIVE_TEN_K = (5, 10, 13)


def _card_key(card):
    return (card.rank, card.suit or "")


def _hand_key(hand):
    return tuple(sorted(_card_key(c) for c in hand))


@lru_cache(maxsize=None)
def _min_turns_to_empty_key(hand_key):
    if not hand_key:
        return 0

    hand = [Card(rank, suit or None) for rank, suit in hand_key]
    moves = MoveGenerator(hand).generate_all()
    if not moves:
        return len(hand)

    best = len(hand)
    for move in moves:
        remaining = list(hand)
        for c in move:
            remaining.remove(c)
        best = min(best, 1 + _min_turns_to_empty_key(_hand_key(remaining)))
    return best


def _min_turns_to_empty(hand):
    return _min_turns_to_empty_key(_hand_key(hand))


def _can_remove_cards_once(hand, move_cards):
    remaining = list(hand)
    try:
        for c in move_cards:
            remaining.remove(c)
    except ValueError:
        return False, list(hand)
    return True, remaining


def _count_straight_windows(ranks):
    unique = sorted(set(r for r in ranks if r < 15))
    windows = 0
    for i in range(len(unique) - 4):
        seq = unique[i:i + 5]
        if all(seq[j] + 1 == seq[j + 1] for j in range(4)):
            windows += 1
    return windows


def _count_bombs(hand, rank_counts=None):
    counts = rank_counts if rank_counts is not None else Counter(c.rank for c in hand)
    bombs = sum(1 for cnt in counts.values() if cnt == 4)
    if 20 in counts and 30 in counts:
        bombs += 1
    if 5 in counts and 10 in counts and 13 in counts:
        bombs += 1
    return bombs


def _best_rank_with_count(counts, threshold):
    eligible = [rank for rank, cnt in counts.items() if cnt >= threshold]
    return max(eligible) if eligible else 0


def _expected_opp_penalty(infoset):
    if infoset is None:
        return 0.0
    unseen_counts = infoset.get("unseen_counts", [])
    if not unseen_counts:
        return 0.0
    total_unseen_cards = float(sum(unseen_counts))
    if total_unseen_cards <= 0:
        return 0.0
    total_unseen_points = 0.0
    for idx, rank in enumerate(RANKS):
        total_unseen_points += float(unseen_counts[idx]) * POINT_VALUES.get(rank, 0)
    opp_cards = float(infoset.get("opp_card_count", 0))
    return opp_cards * total_unseen_points / total_unseen_cards


def _endgame_features(hand, infoset=None):
    if not hand:
        opp_expected_penalty = _expected_opp_penalty(infoset)
        swing_if_end_now = opp_expected_penalty
        return np.array([
            0.0,
            opp_expected_penalty / 50.0,
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            swing_if_end_now / 50.0,
        ], dtype=np.float32)

    counts = Counter(c.rank for c in hand)
    ranks = [c.rank for c in hand]
    self_penalty = float(sum(POINT_VALUES.get(c.rank, 0) for c in hand))
    opp_expected_penalty = _expected_opp_penalty(infoset)
    min_turns = _min_turns_to_empty(hand)
    best_single = max(ranks) if ranks else 0
    best_pair = _best_rank_with_count(counts, 2)
    best_triple = _best_rank_with_count(counts, 3)
    lowest_single = min(ranks) if ranks else 0
    swing_if_end_now = opp_expected_penalty - self_penalty

    return np.array([
        self_penalty / 50.0,
        opp_expected_penalty / 50.0,
        min_turns / 5.0,
        1.0 if min_turns <= 2 else 0.0,
        1.0 if min_turns <= 3 else 0.0,
        best_single / 30.0,
        best_pair / 30.0,
        best_triple / 30.0,
        lowest_single / 30.0,
        swing_if_end_now / 50.0,
    ], dtype=np.float32)


def _hand_summary(hand):
    if not hand:
        return np.zeros(HAND_SUMMARY_DIM, dtype=np.float32)

    ranks = [c.rank for c in hand]
    counts = Counter(ranks)
    point_total = sum(POINT_VALUES.get(c.rank, 0) for c in hand)
    point_card_count = sum(1 for c in hand if c.rank in POINT_VALUES)
    high_single_count = sum(1 for rank, cnt in counts.items() if cnt == 1 and rank >= 13)
    high_card_count = sum(1 for c in hand if c.rank >= 13)
    control_card_count = sum(1 for c in hand if c.rank >= 14)
    pair_count = sum(1 for cnt in counts.values() if cnt >= 2)
    triple_count = sum(1 for cnt in counts.values() if cnt >= 3)
    quad_count = sum(1 for cnt in counts.values() if cnt >= 4)
    bomb_count = _count_bombs(hand, counts)
    straight_windows = _count_straight_windows(ranks)
    distinct_rank_count = len(counts)
    max_rank = max(ranks)

    return np.array([
        len(hand) / 5.0,
        point_total / 50.0,  # a 5-card hand can hold up to ~50 points (4 K + 10)
        point_card_count / 5.0,
        high_single_count / 5.0,
        high_card_count / 5.0,
        control_card_count / 5.0,
        pair_count / 3.0,
        triple_count / 2.0,
        quad_count / 1.0,
        bomb_count / 2.0,
        straight_windows / 2.0,
        distinct_rank_count / 5.0,
        max_rank / 30.0,
    ], dtype=np.float32)


def _move_consequence_features(move, infoset, remaining_hand, pre_counts, post_counts):
    if move is None or infoset is None:
        return np.zeros(CONSEQUENCE_DIM, dtype=np.float32)

    move_cards = move.cards if isinstance(move, Move) else move
    point_total_before = max(1, sum(POINT_VALUES.get(c.rank, 0) for c in infoset["hand"]))
    point_spent = sum(POINT_VALUES.get(c.rank, 0) for c in move_cards) / point_total_before

    breaks_pair = 0.0
    breaks_triple = 0.0
    breaks_quad = 0.0
    for rank, before in pre_counts.items():
        after = post_counts.get(rank, 0)
        if 0 < after < before:
            if before >= 2:
                breaks_pair = 1.0
            if before >= 3:
                breaks_triple = 1.0
            if before >= 4:
                breaks_quad = 1.0

    pre_bombs = _count_bombs(infoset["hand"], pre_counts)
    post_bombs = _count_bombs(remaining_hand, post_counts)
    reduces_bomb_count = 1.0 if post_bombs < pre_bombs else 0.0

    max_rank = max(pre_counts.keys()) if pre_counts else 0
    spends_top_singleton = 0.0
    for c in move_cards:
        if c.rank == max_rank and pre_counts[c.rank] == 1:
            spends_top_singleton = 1.0
            break

    return np.array([
        point_spent,
        breaks_pair,
        breaks_triple,
        breaks_quad,
        reduces_bomb_count,
        spends_top_singleton,
    ], dtype=np.float32)


def _action_extra_features(move):
    """Explicit card-economy signals for a candidate move (targets high-card waste),
    plus within-tier bomb strength (suit for a suited 5-10-K, rank for a quad)."""
    if move is None:
        return np.zeros(ACTION_EXTRA_DIM, dtype=np.float32)
    m = move if isinstance(move, Move) else Move(move)
    cards = m.cards
    control_spent = sum(1 for c in cards if c.rank >= CONTROL_RANK)
    point_spent = sum(1 for c in cards if c.rank in POINT_VALUES)
    spends_joker = 1.0 if any(c.rank >= 20 for c in cards) else 0.0
    # strength[0] (the bomb tier) is already encoded in the move-context block; this
    # adds strength[1], the within-tier tiebreak that distinguishes e.g. a quad of 3s
    # from a quad of 2s, or a spade 5-10-K from a diamond one. 0 for non-bombs.
    bomb_tiebreak = (m.strength[1] / 13.0) if (m.type == "bomb" and len(m.strength) > 1) else 0.0
    return np.array([control_spent / 4.0, point_spent / 4.0, spends_joker, bomb_tiebreak], dtype=np.float32)


def _510k_suit_features(hand):
    """Suit-awareness for the 5-10-K bomb. Rank-only features cannot tell a *suited*
    5-10-K (tier-2 bomb, beats almost everything) from an *offsuit* one (tier-1, the
    weakest bomb) -- yet the game does (game.py:124-131). This block exposes that."""
    if not hand:
        return np.zeros(SUIT_510K_DIM, dtype=np.float32)
    suit_ranks = {}
    have = {5: False, 10: False, 13: False}
    for c in hand:
        if c.rank in have:
            have[c.rank] = True
            if c.suit:
                suit_ranks.setdefault(c.suit, set()).add(c.rank)
    suited_full = sum(1 for rs in suit_ranks.values() if rs == set(FIVE_TEN_K))
    best_progress = max((len(rs) for rs in suit_ranks.values()), default=0)  # 0..3
    can_form = 1.0 if all(have.values()) else 0.0
    offsuit_only = 1.0 if (can_form and suited_full == 0) else 0.0
    return np.array([
        suited_full / 2.0,    # strong (suited) 5-10-K bombs held
        best_progress / 3.0,  # closeness to assembling a suited bomb
        can_form,             # can assemble any 5-10-K bomb at all
        offsuit_only,         # only the weak offsuit version is available
    ], dtype=np.float32)


def _tempo_endgame_features(infoset):
    """Draw-priority and go-out-value signals that rank-only state features miss.

    - Draw priority: the trick winner refills first (game.py:359), so when the deck
      is nearly empty the last-trick winner enters the endgame with a card edge.
    - Go-out value: emptying your hand once the deck is gone wins pot + a flat +20
      bonus + the opponent's leftover point penalty (game.py:561-567). The +20
      magnitude is otherwise invisible to the features.
    All inputs here are public (no opponent-hand leak)."""
    if infoset is None:
        return np.zeros(TEMPO_DIM, dtype=np.float32)
    deck = float(infoset.get("deck_size", 0))
    is_endgame = deck == 0
    me = infoset.get("player_index", 0)
    last_winner = infoset.get("last_hand_winner")
    priority = 0.0 if last_winner is None else (1.0 if last_winner == me else -1.0)
    low_deck = 1.0 if 0 < deck <= 5 else 0.0
    can_empty = float(infoset.get("can_empty_hand", 0))
    pot = float(infoset.get("current_pot", 0))
    go_out_value = ((pot + 20.0 + _expected_opp_penalty(infoset)) / 60.0) if (is_endgame and can_empty) else 0.0
    return np.array([
        priority,                 # who won the last trick (+1 me / -1 opp / 0 none)
        priority * low_deck,      # draw-priority edge as the deck dries up
        go_out_value,             # value of going out right now (incl. the +20)
    ], dtype=np.float32)


def _decisiveness_features(infoset):
    """How decided the game is. With ~100 points in play, a lead bigger than what the
    opponent can still swing (the points left to contest + the 20-point go-out bonus)
    is an unlosable position -- the emergent '>60 = can't lose' mechanic. Lets the
    policy learn to *stop spending resources once the win is secured* and to race for
    the threshold when it's close, instead of grabbing points it no longer needs."""
    if infoset is None:
        return np.zeros(DECISIVE_DIM, dtype=np.float32)
    sp = float(infoset.get("self_points", 0))
    op = float(infoset.get("opp_points", 0))
    remaining = max(0.0, 100.0 - sp - op)      # points not yet locked into a score
    max_swing = remaining + 20.0               # opp wins everything left + the go-out bonus
    ratio = (sp - op) / (max_swing + 1.0)      # |ratio| > 1  ==>  the game is locked
    return np.array([
        max(-2.0, min(2.0, ratio)) / 2.0,
        1.0 if ratio > 1.0 else 0.0,           # win locked
        1.0 if ratio < -1.0 else 0.0,          # loss locked
    ], dtype=np.float32)


def _belief_prior_features(infoset):
    """Cheap, hand-derived priors over what the opponent could be holding."""
    if infoset is None:
        return np.zeros(BELIEF_PRIOR_DIM, dtype=np.float32)
    unseen = infoset.get("unseen_counts", [])
    unseen_control = 0.0
    unseen_point_value = 0.0
    for idx, r in enumerate(RANKS):
        u = float(unseen[idx]) if idx < len(unseen) else 0.0
        if r >= CONTROL_RANK:
            unseen_control += u
        unseen_point_value += u * POINT_VALUES.get(r, 0)
    total_unseen = float(sum(unseen)) if unseen else 0.0
    opp_cards = float(infoset.get("opp_card_count", 0))
    my_cards = float(infoset.get("hand_size", 0))
    exp_opp_control = (opp_cards * unseen_control / total_unseen) if total_unseen > 0 else 0.0
    exp_opp_points = (opp_cards * unseen_point_value / total_unseen) if total_unseen > 0 else 0.0
    return np.array([
        unseen_control / TOTAL_CONTROL_CARDS,
        unseen_point_value / 100.0,
        total_unseen / 30.0,
        exp_opp_control / 5.0,
        exp_opp_points / 30.0,
        (my_cards - opp_cards) / 5.0,
        infoset.get("self_points", 0) / 100.0,
        infoset.get("opp_points", 0) / 100.0,
    ], dtype=np.float32)


def _max_unseen_rank(infoset):
    unseen = infoset.get("unseen_counts", [])
    mx = 0
    for i, r in enumerate(RANKS):
        if i < len(unseen) and unseen[i] > 0 and r > mx:
            mx = r
    return mx


def _boss_features(infoset):
    """'Boss' cards currently outrank every card still unseen -> guaranteed to win a
    led trick (modulo bombs). This is the option value the policy was missing: it can
    now *see* that its 2 is the highest card left, so spending it on nothing is bad."""
    if infoset is None:
        return np.zeros(BOSS_DIM, dtype=np.float32)
    hand = infoset["hand"]
    if not hand:
        return np.zeros(BOSS_DIM, dtype=np.float32)
    mur = _max_unseen_rank(infoset)
    ranks = [c.rank for c in hand]
    my_max = max(ranks)
    n_boss = sum(1 for r in ranks if r > mur)
    unseen = infoset.get("unseen_counts", [])
    above_top = sum(unseen[i] for i, r in enumerate(RANKS) if r > my_max and i < len(unseen))
    return np.array([
        min(1.0, n_boss / 4.0),
        1.0 if my_max > mur else 0.0,
        min(1.0, above_top / 6.0),
    ], dtype=np.float32)


def _510k_potential(hand):
    suit_ranks = {}
    have = {5: False, 10: False, 13: False}
    for c in hand:
        if c.rank in have:
            have[c.rank] = True
            if c.suit:
                suit_ranks.setdefault(c.suit, set()).add(c.rank)
    suited = sum(1 for rs in suit_ranks.values() if rs == set(FIVE_TEN_K))
    return suited + (0.5 if all(have.values()) else 0.0)


def _move_vulnerability(m, infoset):
    """Rough P(opponent can beat this non-bomb move), from the unseen distribution and
    opp hand size, plus the points it risks. Teaches the policy that a point-pair is
    only safe when the opponent likely can't top it."""
    if m is None or m.type == "bomb":
        return 0.0, 0.0
    unseen = infoset.get("unseen_counts", [])
    total = float(sum(unseen)) or 1.0
    opp = float(infoset.get("opp_card_count", 0))
    of = min(1.0, opp / total)
    r = m.strength[0]
    counts = {RANKS[i]: (unseen[i] if i < len(unseen) else 0) for i in range(len(RANKS))}
    if m.type in ("pair", "triple"):
        k = len(m.cards)
        p_none = 1.0
        for rank, u in counts.items():
            if rank > r and u >= k:
                p_none *= (1.0 - of ** k)
        p_same = 1.0 - p_none
    else:  # single / straight: beaten by any higher card of the type
        num_higher = sum(u for rank, u in counts.items() if rank > r)
        p_same = 1.0 - (1.0 - of) ** num_higher
    bomb = 0.0
    if counts.get(20, 0) >= 1 and counts.get(30, 0) >= 1:
        bomb = max(bomb, of * of)
    for rank, u in counts.items():
        if u >= 4:
            bomb = max(bomb, of ** 4)
    if counts.get(5, 0) >= 1 and counts.get(10, 0) >= 1 and counts.get(13, 0) >= 1:
        bomb = max(bomb, of ** 3)
    vuln = min(1.0, p_same + 0.4 * bomb)
    pts = sum(POINT_VALUES.get(c.rank, 0) for c in m.cards)
    return vuln, min(1.0, vuln * pts / 25.0)


def _action_risk_features(move, infoset, remaining_hand):
    """Per-move: does it spend a boss card, how likely is it beaten, points at risk,
    and does it destroy 5-10-K bomb material."""
    if move is None or infoset is None:
        return np.zeros(ACTION_RISK_DIM, dtype=np.float32)
    m = move if isinstance(move, Move) else Move(move)
    mur = _max_unseen_rank(infoset)
    spends_boss = 1.0 if any(c.rank > mur for c in m.cards) else 0.0
    vuln, at_risk = _move_vulnerability(m, infoset)
    reduces_510k = max(0.0, min(1.0, _510k_potential(infoset["hand"]) - _510k_potential(remaining_hand)))
    return np.array([spends_boss, vuln, at_risk, reduces_510k], dtype=np.float32)


def move_feature_dim():
    return (len(MOVE_TYPES) + MOVE_CONTEXT_DIM + HAND_SUMMARY_DIM + ACTION_ENDGAME_DIM
            + CONSEQUENCE_DIM + len(RANKS) + ACTION_EXTRA_DIM)


def encode_move(move, infoset=None):
    vec = np.zeros(move_feature_dim(), dtype=np.float32)
    pre_counts = Counter(c.rank for c in infoset["hand"]) if infoset is not None else Counter()
    remaining_hand = list(infoset["hand"]) if infoset is not None else []
    post_counts = pre_counts.copy()
    can_score_consequences = False

    if move is None:
        vec[MOVE_TO_IDX["none"]] = 1.0
    else:
        m = move if isinstance(move, Move) else Move(move)
        vec[MOVE_TO_IDX[m.type]] = 1.0
        if infoset is not None:
            can_score_consequences, remaining_hand = _can_remove_cards_once(infoset["hand"], m.cards)
            if can_score_consequences:
                post_counts = Counter(c.rank for c in remaining_hand)
            else:
                remaining_hand = list(infoset["hand"])
                post_counts = pre_counts.copy()

    base = len(MOVE_TYPES)

    if move is None:
        vec[base + 0] = 0.0
        vec[base + 1] = 0.0
        vec[base + 2] = 0.0
        vec[base + 3] = 0.0
        vec[base + 4] = 0.0
        vec[base + 5] = 0.0
    else:
        m = move if isinstance(move, Move) else Move(move)
        pts = sum(POINT_VALUES.get(c.rank, 0) for c in m.cards)
        vec[base + 0] = len(m.cards) / 5.0
        vec[base + 1] = pts / 40.0
        vec[base + 2] = (m.strength[0] / 30.0) if m.type != "bomb" else (m.strength[0] / 4.0)
        vec[base + 3] = 1.0 if m.type == "bomb" else 0.0
        vec[base + 4] = 1.0 if (len(m.cards) == 1 and m.cards[0].rank >= 13) else 0.0
        hand_size = max(1, infoset.get("hand_size", len(m.cards))) if infoset is not None else len(m.cards)
        vec[base + 5] = 1.0 if can_score_consequences and len(m.cards) == hand_size else 0.0

    if infoset is not None:
        vec[base + 6] = 1.0 if infoset.get("opp_about_to_win", 0) else 0.0
        vec[base + 7] = infoset.get("current_pot", 0) / 40.0
        vec[base + 8] = infoset.get("opp_card_count", 5) / 5.0

    base += MOVE_CONTEXT_DIM
    vec[base:base + HAND_SUMMARY_DIM] = _hand_summary(remaining_hand)
    base += HAND_SUMMARY_DIM
    vec[base:base + ACTION_ENDGAME_DIM] = _endgame_features(remaining_hand, infoset)
    base += ACTION_ENDGAME_DIM
    if can_score_consequences:
        vec[base:base + CONSEQUENCE_DIM] = _move_consequence_features(move, infoset, remaining_hand, pre_counts, post_counts)
    base += CONSEQUENCE_DIM

    if move is not None:
        m = move if isinstance(move, Move) else Move(move)
        for c in m.cards:
            vec[base + RANK_TO_IDX[c.rank]] += 1.0
    base += len(RANKS)

    vec[base:base + ACTION_EXTRA_DIM] = _action_extra_features(move)

    return vec


def encode_state(infoset):
    hand = infoset["hand"]
    played = infoset["played_cards"]
    unseen = infoset["unseen_counts"]

    hand_counts = np.zeros(len(RANKS), dtype=np.float32)
    for c in hand:
        hand_counts[RANK_TO_IDX[c.rank]] += 1

    played_counts = np.zeros(len(RANKS), dtype=np.float32)
    for c in played:
        played_counts[RANK_TO_IDX[c.rank]] += 1

    unseen_counts = np.array(unseen, dtype=np.float32)

    structure = np.array([
        np.sum(hand_counts == 1),
        np.sum(hand_counts == 2),
        np.sum(hand_counts == 3),
        np.sum(hand_counts == 4),
    ], dtype=np.float32)

    control = np.array([
        float(infoset["last_action_was_pass"]),
        infoset["pass_count"] / 2.0,
        infoset["opp_card_count"] / 5.0,
    ], dtype=np.float32)

    global_state = np.array([
        infoset["deck_size"] / 54.0,
        infoset["hand_size"] / 5.0,
        infoset["last_hand_points"] / 40.0,
        infoset.get("current_pot", 0) / 40.0,
        infoset["self_points"] / 100.0,
        infoset["opp_points"] / 100.0,
    ], dtype=np.float32)

    last_move = encode_move(infoset["last_move"], infoset)

    hand_type = np.zeros(len(MOVE_TYPES), dtype=np.float32)
    if infoset["hand_type"] is None:
        hand_type[MOVE_TO_IDX["none"]] = 1.0
    else:
        hand_type[MOVE_TO_IDX[infoset["hand_type"]]] = 1.0

    hand_summary = _hand_summary(hand)
    endgame = _endgame_features(hand, infoset)
    suit_510k = _510k_suit_features(hand)
    tempo = _tempo_endgame_features(infoset)
    belief_prior = _belief_prior_features(infoset)

    extras = np.array([
        float(infoset["can_empty_hand"]),
        infoset["num_move_types"] / 5.0,
        float(infoset["has_control"]),
        float(infoset["opp_about_to_win"]),
        float(infoset["is_endgame"]),
        infoset["point_diff"] / 50.0,
        float(infoset.get("last_hand_winner_is_self", 0)),
    ], dtype=np.float32)

    return np.concatenate([
        hand_counts,
        played_counts,
        unseen_counts,
        structure,
        control,
        global_state,
        last_move,
        hand_type,
        hand_summary,
        endgame,
        suit_510k,
        tempo,
        belief_prior,
        extras,
    ]).astype(np.float32)
