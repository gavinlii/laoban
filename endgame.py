"""
Exact endgame solver for Five-Ten-K.

Key fact: once the deck is empty, the game is *perfect information*. Every card is
in someone's hand or has been played, so the cards a player hasn't seen are exactly
the opponent's hand (deducible from the played pile). The endgame is therefore a
small, perfect-information, zero-sum game and can be solved EXACTLY by minimax.

This both (a) lets the deployed bot play the endgame optimally (fixing the weak
run-out / tempo play), and (b) provides exact value targets at deck-exhaustion for
training, so the policy learns the true value of conserving bombs/high cards for
the endgame -- without any reward hacking.

The solver mirrors GameEnv's deck-empty rules exactly:
  * a play adds its point cards to the pot and is removed from hand;
  * emptying your hand wins  pot + 20 + (opponent's leftover point penalty);
  * a pass ends the trick, awarding the pot to the last player to have played,
    who then leads the next trick (no draws -- the deck is empty).
"""
from collections import Counter
from functools import lru_cache

from game import Card, Move, MoveGenerator, POINT_VALUES


def _pts(cards):
    return sum(POINT_VALUES.get(c.rank, 0) for c in cards)


def _hand_points(hand_key):
    return sum(POINT_VALUES.get(r, 0) for r, _ in hand_key)


def _key(hand):
    return tuple(sorted((c.rank, c.suit or "") for c in hand))


def _cards(hand_key):
    return [Card(r, s or None) for r, s in hand_key]


def _move_strength_repr(move_key):
    if move_key is None:
        return None
    m = Move(_cards(move_key))
    return (m.type, m.strength, m.length)


@lru_cache(maxsize=None)
def _legal(hand_key, hand_type, last_repr):
    """Mirror GameEnv.get_legal_actions for the deck-empty case. Returns a tuple of
    move keys (each a tuple of (rank,suit)), plus None for pass when following."""
    hand = _cards(hand_key)
    moves = MoveGenerator(hand).generate_all()
    last_type = last_repr[0] if last_repr else None
    last_strength = last_repr[1] if last_repr else None
    last_length = last_repr[2] if last_repr else None
    legal = []
    for m in moves:
        mv = Move(m)
        if hand_type is None:
            legal.append(tuple(sorted((c.rank, c.suit or "") for c in m)))
        elif mv.type == "bomb":
            if last_repr is None or last_type != "bomb" or mv.strength > last_strength:
                legal.append(tuple(sorted((c.rank, c.suit or "") for c in m)))
        elif last_repr is not None and last_type == "bomb":
            continue
        elif mv.type == hand_type:
            if mv.type == "straight":
                if mv.length == last_length and mv.strength > last_strength:
                    legal.append(tuple(sorted((c.rank, c.suit or "") for c in m)))
            elif mv.strength > last_strength:
                legal.append(tuple(sorted((c.rank, c.suit or "") for c in m)))
    if hand_type is not None:
        legal.append(None)
    return tuple(legal)


def _remove(hand_key, move_key):
    remaining = list(hand_key)
    for c in move_key:
        remaining.remove(c)
    return tuple(remaining)


def endgame_value(hands, to_move, ref_seat, hand_type=None, last_move=None, last_player=None, pot=0, memo=None):
    """Optimal future margin (ref_seat minus opponent) from a deck-empty position,
    under optimal play by both sides. `hands` is {0: list[Card], 1: list[Card]};
    `last_move` is a move key (tuple of (rank,suit)) or None."""
    h0, h1 = _key(hands[0]), _key(hands[1])
    last_repr = _move_strength_repr(last_move)
    return _solve(h0, h1, to_move, ref_seat, hand_type, last_repr, last_player, pot,
                  {} if memo is None else memo)


def endgame_value_from_env(env, ref_seat):
    """Exact endgame value (ref_seat margin) from a god-view GameEnv with deck empty."""
    last_move = env.last_move
    last_key = _key(last_move.cards) if last_move is not None else None
    return endgame_value(
        env.hands, env.current_player, ref_seat,
        hand_type=env.hand_type, last_move=last_key,
        last_player=env.last_player, pot=env.current_pot, memo={},
    )


def _solve(h0, h1, to_move, ref_seat, hand_type, last_repr, last_player, pot, memo):
    state = (h0, h1, to_move, hand_type, last_repr, last_player, pot)
    cached = memo.get(state)
    if cached is not None:
        return cached

    hands = (h0, h1)
    legal = _legal(hands[to_move], hand_type, last_repr)
    best = None
    maximizing = (to_move == ref_seat)

    for mk in legal:
        if mk is None:
            # pass -> trick ends; pot to last_player, who leads the next trick
            winner = last_player
            delta = pot if winner == ref_seat else -pot
            child = _solve(h0, h1, winner, ref_seat, None, None, None, 0, memo)
            val = delta + child
        else:
            mv = Move(_cards(mk))
            gained = _pts(_cards(mk))
            new_pot = pot + gained
            if to_move == 0:
                nh0, nh1 = _remove(h0, mk), h1
            else:
                nh0, nh1 = h0, _remove(h1, mk)
            mover_hand = nh0 if to_move == 0 else nh1
            if len(mover_hand) == 0:
                # to_move empties hand: wins pot + 20 + opponent leftover penalty
                loser_hand = nh1 if to_move == 0 else nh0
                swing = new_pot + 20 + _hand_points(loser_hand)
                val = swing if to_move == ref_seat else -swing
            else:
                nht = hand_type if hand_type is not None else mv.type
                val = _solve(nh0, nh1, 1 - to_move, ref_seat,
                             nht, _move_strength_repr(mk), to_move, new_pot, memo)
        if best is None:
            best = val
        elif maximizing:
            best = max(best, val)
        else:
            best = min(best, val)

    memo[state] = best
    return best


def best_action(hands, to_move, hand_type=None, last_move=None, last_player=None, pot=0, memo=None):
    """The optimal move (list[Card] or None for pass) for `to_move`, returned as actual
    Card objects from its hand. Evaluates each move for `to_move` (ref = to_move) and
    picks the one maximizing its own final margin."""
    memo = {} if memo is None else memo
    h0, h1 = _key(hands[0]), _key(hands[1])
    last_repr = _move_strength_repr(last_move)
    ref = to_move
    legal = _legal((h0, h1)[to_move], hand_type, last_repr)
    best_val, best_mk = None, None
    for mk in legal:
        if mk is None:
            winner = last_player
            delta = pot if winner == ref else -pot
            val = delta + _solve(h0, h1, winner, ref, None, None, None, 0, memo)
        else:
            new_pot = pot + _pts(_cards(mk))
            if to_move == 0:
                nh0, nh1 = _remove(h0, mk), h1
            else:
                nh0, nh1 = h0, _remove(h1, mk)
            mover_hand = nh0 if to_move == 0 else nh1
            if len(mover_hand) == 0:
                loser_hand = nh1 if to_move == 0 else nh0
                val = new_pot + 20 + _hand_points(loser_hand)
            else:
                mv = Move(_cards(mk))
                nht = hand_type if hand_type is not None else mv.type
                val = _solve(nh0, nh1, 1 - to_move, ref, nht, _move_strength_repr(mk), to_move, new_pot, memo)
        if best_val is None or val > best_val:
            best_val, best_mk = val, mk
    if best_mk is None:
        return None
    return _match_cards(hands[to_move], best_mk)


def _match_cards(hand, move_key):
    chosen, used = [], []
    for rank, suit in move_key:
        for c in hand:
            if c.rank == rank and (c.suit or "") == suit and id(c) not in used:
                chosen.append(c)
                used.append(id(c))
                break
    return chosen


class EndgameSolverPlayer:
    """Plays optimally once the deck is empty (perfect information), and defers to a
    fallback player before that. Used to give the deployed bot an exact endgame."""

    wants_concrete_same_rank_choices = False

    def __init__(self, fallback_player):
        self.fallback = fallback_player

    def act(self, infoset):
        if infoset.get("deck_size", 1) != 0:
            return self.fallback.act(infoset)
        return endgame_act_from_infoset(infoset)


def endgame_act_from_infoset(infoset):
    """At the endgame the opponent's hand == the unseen cards, so we can solve exactly."""
    from train_ppo import ALL_CARDS  # the full 54-card list
    me = infoset["player_index"]
    seen = set((c.rank, c.suit or "") for c in infoset["hand"])
    for c in infoset["played_cards"]:
        seen.add((c.rank, c.suit or ""))
    opp_hand = [c for c in ALL_CARDS if (c.rank, c.suit or "") not in seen]
    hands = {me: list(infoset["hand"]), 1 - me: opp_hand}
    last_move = infoset.get("last_move")
    last_move_key = _key(last_move.cards) if last_move is not None else None
    return best_action(
        hands, me,
        hand_type=infoset.get("hand_type"),
        last_move=last_move_key,
        last_player=infoset.get("last_player"),
        pot=infoset.get("current_pot", 0),
        memo={},
    )
