import random
from collections import Counter, deque
from copy import deepcopy

# ======================
# Constants
# ======================

RANKS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 20, 30]
SUITS = ['H', 'D', 'C', 'S']
SUIT_ORDER = {'D': 0, 'C': 1, 'H': 2, 'S': 3}

SMALL_JOKER = 20
BIG_JOKER = 30

POINT_VALUES = {5: 5, 10: 10, 13: 10}

# Total count per rank (for unseen calculation)
RANK_TOTALS = {
    r: (1 if r in [SMALL_JOKER, BIG_JOKER] else 4)
    for r in RANKS
}

# Bomb quad rank order: 3 < 4 < ... < K < A < 2
BOMB_QUAD_RANK_ORDER = {
    3: 0,
    4: 1,
    5: 2,
    6: 3,
    7: 4,
    8: 5,
    9: 6,
    10: 7,
    11: 8,   # J
    12: 9,   # Q
    13: 10,  # K
    14: 11,  # A
    17: 12,  # 2
}

HISTORY_MAXLEN = 96


# ======================
# Card
# ======================

class Card:
    def __init__(self, rank, suit=None):
        self.rank = rank
        self.suit = suit

    def __repr__(self):
        if self.rank == SMALL_JOKER:
            return "X"
        if self.rank == BIG_JOKER:
            return "D"
        return f"{self.rank}{self.suit}"

    def __eq__(self, other):
        return isinstance(other, Card) and self.rank == other.rank and self.suit == other.suit

    def __hash__(self):
        return hash((self.rank, self.suit))


# ======================
# Deck
# ======================

class Deck:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.cards = self._build()
        self.rng.shuffle(self.cards)

    def _build(self):
        cards = []
        for r in RANKS:
            if r in [SMALL_JOKER, BIG_JOKER]:
                cards.append(Card(r))
            else:
                for s in SUITS:
                    cards.append(Card(r, s))
        return cards

    def draw(self):
        return self.cards.pop() if self.cards else None

    def size(self):
        return len(self.cards)


# ======================
# Move
# ======================

class Move:
    def __init__(self, cards):
        self.cards = sorted(cards, key=lambda c: c.rank)
        self.type = None
        self.strength = None
        self.length = len(cards)
        self._analyze()

    def _analyze(self):
        ranks = [c.rank for c in self.cards]
        count = Counter(ranks)

        # Joker bomb
        if set(ranks) == {SMALL_JOKER, BIG_JOKER} and len(ranks) == 2:
            self.type = "bomb"
            self.strength = (4, 0)
            return

        # Four of a kind
        if 4 in count.values():
            quad_rank = next(rank for rank, cnt in count.items() if cnt == 4)
            self.type = "bomb"
            self.strength = (3, BOMB_QUAD_RANK_ORDER[quad_rank])
            return

        # 5-10-K bomb
        if sorted(ranks) == [5, 10, 13]:
            suited = len(set(c.suit for c in self.cards)) == 1
            self.type = "bomb"
            if suited:
                suit = self.cards[0].suit
                self.strength = (2, SUIT_ORDER[suit])
            else:
                self.strength = (1, 0)
            return

        # Normal
        if len(self.cards) == 1:
            self.type = "single"
            self.strength = (ranks[0],)

        elif len(self.cards) == 2 and len(count) == 1:
            self.type = "pair"
            self.strength = (ranks[0],)

        elif len(self.cards) == 3 and len(count) == 1:
            self.type = "triple"
            self.strength = (ranks[0],)

        elif self._is_straight(ranks):
            self.type = "straight"
            self.strength = (max(ranks), 5)

    def _is_straight(self, ranks):
        if len(ranks) != 5:
            return False
        ranks = sorted(ranks)
        if any(r >= 15 for r in ranks):
            return False
        return all(ranks[i] + 1 == ranks[i + 1] for i in range(4))


# ======================
# Move Generator
# ======================

class MoveGenerator:
    def __init__(self, hand):
        self.hand = sorted(hand, key=lambda c: (c.rank, c.suit or ''))

    def generate_all(self, concrete_same_rank_choices=False):
        moves = []

        # singles
        for c in self.hand:
            moves.append([c])

        # group by rank
        rank_map = {}
        for c in self.hand:
            rank_map.setdefault(c.rank, []).append(c)

        # pairs/triples/quads
        for cards in rank_map.values():
            if concrete_same_rank_choices:
                if len(cards) >= 2:
                    moves.extend(self._combinations(cards, 2))
                if len(cards) >= 3:
                    moves.extend(self._combinations(cards, 3))
                if len(cards) >= 4:
                    moves.extend(self._combinations(cards, 4))
            else:
                if len(cards) >= 2:
                    moves.append(cards[:2])
                if len(cards) >= 3:
                    moves.append(cards[:3])
                if len(cards) >= 4:
                    moves.append(cards[:4])

        # 5-10-K bombs
        for combo in self._combinations(self.hand, 3):
            if set(c.rank for c in combo) == {5, 10, 13}:
                moves.append(combo)

        # joker bomb
        jokers = [c for c in self.hand if c.rank in [SMALL_JOKER, BIG_JOKER]]
        if len(jokers) == 2:
            moves.append(jokers)

        # straights (length 5)
        unique = sorted(set(c.rank for c in self.hand if c.rank < 15))
        for i in range(len(unique) - 4):
            seq = unique[i:i + 5]
            if all(seq[k] + 1 == seq[k + 1] for k in range(4)):
                move = []
                used = set()
                for r in seq:
                    for c in self.hand:
                        if c.rank == r and c not in used:
                            move.append(c)
                            used.add(c)
                            break
                moves.append(move)

        return moves

    def _combinations(self, arr, k):
        if k == 0:
            return [[]]
        if len(arr) < k:
            return []
        res = []
        for i in range(len(arr)):
            for tail in self._combinations(arr[i + 1:], k - 1):
                res.append([arr[i]] + tail)
        return res


# ======================
# Game Environment
# ======================

class GameEnv:
    def __init__(self, players, seed=None, verbose=True, history_maxlen=HISTORY_MAXLEN):
        self.players = players
        self.seed = seed
        self.verbose = verbose
        self.history_maxlen = history_maxlen
        self.reset()

    def reset(self):
        self.deck = Deck(self.seed)
        self.hands = {0: [], 1: []}
        self.points = {0: 0, 1: 0}
        self.done = False

        # tracking across game
        self.played_cards = []
        self.played_rank_counts = {r: 0 for r in RANKS}
        self.last_hand_points = 0
        self.last_hand_winner = None
        self.face_up = {}
        self.public_history = deque(maxlen=self.history_maxlen)

        # deal
        for _ in range(5):
            for p in [0, 1]:
                self.hands[p].append(self.deck.draw())

        self.face_up = {p: random.choice(self.hands[p]) for p in [0, 1]}
        self.current_player = 0 if self.face_up[0].rank > self.face_up[1].rank else 1

        self._start_new_hand(self.current_player)
        self._record_event({
            "kind": "deal",
            "actor": -1,
            "deck_size": self.deck.size(),
            "hand_sizes": [len(self.hands[0]), len(self.hands[1])],
            "current_player": self.current_player,
            "pot": 0,
            "points": [self.points[0], self.points[1]],
        })

    def _start_new_hand(self, starting_player):
        self.hand_type = None
        self.last_move = None
        self.last_player = None
        self.last_action_was_pass = False
        self.pass_count = 0
        self.current_pot = 0
        self.current_player = starting_player

    def _record_event(self, event):
        e = dict(event)
        if "deck_size" not in e:
            e["deck_size"] = self.deck.size()
        if "hand_sizes" not in e:
            e["hand_sizes"] = [len(self.hands[0]), len(self.hands[1])]
        if "pot" not in e:
            e["pot"] = self.current_pot
        if "points" not in e:
            e["points"] = [self.points[0], self.points[1]]
        self.public_history.append(e)

    def _remove_cards(self, player, cards):
        for c in cards:
            self.hands[player].remove(c)

    def _same_rank_candidates(self, player, rank, k):
        cards = [c for c in self.hands[player] if c.rank == rank]
        if len(cards) < k:
            return []
        return MoveGenerator(cards)._combinations(cards, k)

    def _preserved_510k_score(self, remaining_cards):
        suit_ranks = {}
        for c in remaining_cards:
            if c.suit is None:
                continue
            suit_ranks.setdefault(c.suit, set()).add(c.rank)
        full = sum(1 for s in SUITS if {5, 10, 13}.issubset(suit_ranks.get(s, set())))
        partial = sum(len(suit_ranks.get(s, set()) & {5, 10, 13}) for s in SUITS)
        return (full, partial)

    def _resolve_same_rank_action(self, player, action):
        if action is None or len(action) not in (2, 3):
            return action
        ranks = {c.rank for c in action}
        if len(ranks) != 1:
            return action
        rank = next(iter(ranks))
        candidates = self._same_rank_candidates(player, rank, len(action))
        if len(candidates) <= 1:
            return action

        best = None
        best_score = None
        for cand in candidates:
            remaining = list(self.hands[player])
            for c in cand:
                remaining.remove(c)
            score = self._preserved_510k_score(remaining)
            suit_signature = tuple(sorted((c.suit or '') for c in cand))
            total_suit_order = sum(SUIT_ORDER.get(c.suit, -1) for c in cand)
            key = (score[0], score[1], -total_suit_order, suit_signature)
            if best_score is None or key > best_score:
                best_score = key
                best = cand
        return best if best is not None else action

    def _player_wants_concrete_same_rank_choices(self, player_idx):
        return bool(getattr(self.players[player_idx], "wants_concrete_same_rank_choices", False))

    def _count_points(self, cards):
        return sum(POINT_VALUES.get(c.rank, 0) for c in cards)

    def _draw_phase(self, winner):
        order = [winner, 1 - winner]
        draw_counts = {0: 0, 1: 0}
        while self.deck.size() > 0:
            all_full = True
            for p in order:
                if len(self.hands[p]) < 5 and self.deck.size() > 0:
                    self.hands[p].append(self.deck.draw())
                    draw_counts[p] += 1
                    all_full = False
            if all_full:
                break
        self._record_event({
            "kind": "draw",
            "actor": winner,
            "winner": winner,
            "draw_counts": [draw_counts[0], draw_counts[1]],
        })

    def get_legal_actions(self, player, hand_type=None, last_move=None, concrete_same_rank_choices=False):
        hand_type = self.hand_type if hand_type is None else hand_type
        last_move = self.last_move if last_move is None else last_move
        mg = MoveGenerator(self.hands[player])
        moves = mg.generate_all(concrete_same_rank_choices=concrete_same_rank_choices)
        legal = []

        for m in moves:
            move = Move(m)

            if hand_type is None:
                legal.append(m)

            else:
                if move.type == "bomb":
                    if last_move is None or last_move.type != "bomb" or move.strength > last_move.strength:
                        legal.append(m)

                elif last_move is not None and last_move.type == "bomb":
                    continue

                elif move.type == hand_type:
                    if move.type == "straight":
                        if move.length == last_move.length and move.strength > last_move.strength:
                            legal.append(m)
                    else:
                        if move.strength > last_move.strength:
                            legal.append(m)

        if hand_type is not None:
            legal.append(None)

        return legal

    def _compute_unseen_counts(self, player):
        unseen = []
        hand_counts = {r: 0 for r in RANKS}
        for c in self.hands[player]:
            hand_counts[c.rank] += 1
        for r in RANKS:
            unseen.append(RANK_TOTALS[r] - hand_counts[r] - self.played_rank_counts[r])
        return unseen

    def _hidden_targets(self, player):
        opp = 1 - player
        opp_counts = [0] * len(RANKS)
        for c in self.hands[opp]:
            opp_counts[RANKS.index(c.rank)] += 1
        opp_min_turns = self._min_turns_to_empty(self.hands[opp])
        return {
            "opp_rank_counts": opp_counts,
            "opp_point_total": self._count_points(self.hands[opp]),
            "opp_has_bomb": int(self._count_bombs(self.hands[opp]) > 0),
            "opp_can_empty_1": int(opp_min_turns <= 1),
            "opp_can_empty_2": int(opp_min_turns <= 2),
            "opp_min_turns": opp_min_turns,
        }

    def _count_bombs(self, hand):
        counts = Counter(c.rank for c in hand)
        bombs = sum(1 for cnt in counts.values() if cnt == 4)
        if 20 in counts and 30 in counts:
            bombs += 1
        if 5 in counts and 10 in counts and 13 in counts:
            bombs += 1
        return bombs

    def _hand_key(self, hand):
        return tuple(sorted((c.rank, c.suit or '') for c in hand))

    def _min_turns_to_empty(self, hand):
        memo = {}
        def solve(key):
            if not key:
                return 0
            if key in memo:
                return memo[key]
            cards = [Card(r, s or None) for r, s in key]
            moves = MoveGenerator(cards).generate_all()
            if not moves:
                memo[key] = len(cards)
                return memo[key]
            best = len(cards)
            for mv in moves:
                rem = list(cards)
                for c in mv:
                    rem.remove(c)
                best = min(best, 1 + solve(tuple(sorted((c.rank, c.suit or '') for c in rem))))
            memo[key] = best
            return best
        return solve(self._hand_key(hand))

    def get_infoset(self, player=None, concrete_same_rank_choices=None):
        p = self.current_player if player is None else player
        opp = 1 - p
        if concrete_same_rank_choices is None:
            concrete_same_rank_choices = self._player_wants_concrete_same_rank_choices(p)
        legal = self.get_legal_actions(p, concrete_same_rank_choices=concrete_same_rank_choices)
        info = {
            "player_index": p,
            "hand": deepcopy(self.hands[p]),
            "legal_actions": legal,
            "last_move": self.last_move,
            "hand_type": self.hand_type,
            "points": deepcopy(self.points),
            "deck_size": self.deck.size(),
            "played_cards": deepcopy(self.played_cards),
            "last_player": self.last_player,
            "last_action_was_pass": self.last_action_was_pass,
            "pass_count": self.pass_count,
            "opp_card_count": len(self.hands[opp]),
            "last_move_strength": self.last_move.strength if self.last_move else None,
            "has_control": int(self.last_player == p or self.last_player is None),
            "opp_about_to_win": int(len(self.hands[opp]) <= 2),
            "is_endgame": int(self.deck.size() == 0),
            "point_diff": self.points[p] - self.points[opp],
            "self_points": self.points[p],
            "opp_points": self.points[opp],
            "last_hand_winner_is_self": int(self.last_hand_winner == p) if self.last_hand_winner is not None else 0,
            "unseen_counts": self._compute_unseen_counts(p),
            "hand_size": len(self.hands[p]),
            "can_empty_hand": any(a is not None and len(a) == len(self.hands[p]) for a in legal),
            "num_move_types": len(set(Move(a).type for a in legal if a is not None)),
            "last_hand_points": self.last_hand_points,
            "last_hand_winner": self.last_hand_winner,
            "current_pot": self.current_pot,
            "public_history": deepcopy(list(self.public_history)),
        }
        info.update(self._hidden_targets(p))
        return info

    def apply_action(self, action, concrete_same_rank_choices=None):
        if self.done:
            return

        p = self.current_player
        if concrete_same_rank_choices is None:
            concrete_same_rank_choices = self._player_wants_concrete_same_rank_choices(p)
        infoset = self.get_infoset(p, concrete_same_rank_choices=concrete_same_rank_choices)
        if action not in infoset["legal_actions"]:
            raise ValueError(f"Illegal action for player {p}: {action}")

        if not concrete_same_rank_choices:
            action = self._resolve_same_rank_action(p, action)

        if action is None:
            if self.verbose:
                print(f"Player {p} PASS")
            self.last_action_was_pass = True
            self.pass_count += 1
            self._record_event({
                "kind": "pass",
                "actor": p,
                "pass_count": self.pass_count,
                "hand_type": self.hand_type or "none",
            })
        else:
            move = Move(action)
            if self.hand_type is None:
                self.hand_type = move.type
            self.last_player = p
            self.last_action_was_pass = False
            self.pass_count = 0
            self.last_move = move

            gained = self._count_points(action)
            pot_before = self.current_pot
            self.current_pot += gained
            if self.verbose:
                print(f"Player {p} plays {[str(c) for c in action]} (+{gained})")

            self._remove_cards(p, action)
            for c in action:
                self.played_cards.append(c)
                self.played_rank_counts[c.rank] += 1

            self._record_event({
                "kind": "play",
                "actor": p,
                "move_type": move.type,
                "move_rank": move.strength[0] if move.strength else 0,
                "move_len": len(action),
                "points_gained": gained,
                "pot_before": pot_before,
                "pot_after": self.current_pot,
                "is_bomb": int(move.type == "bomb"),
            })

            if self.deck.size() == 0 and len(self.hands[p]) == 0:
                winner = p
                loser = 1 - winner
                loser_penalty = self._count_points(self.hands[loser])
                self.points[winner] += self.current_pot
                self.points[winner] += 20
                self.points[loser] -= loser_penalty
                self.last_hand_points = self.current_pot
                self.last_hand_winner = winner
                self._record_event({
                    "kind": "terminal",
                    "actor": winner,
                    "winner": winner,
                    "hand_points": self.current_pot,
                    "end_bonus": 20,
                    "loser_penalty": loser_penalty,
                })
                self.current_pot = 0
                self.done = True
                self.current_player = winner
                return

        if self.pass_count == 1:
            winner = self.last_player
            if winner is None:
                raise RuntimeError("Hand ended on pass without a previous player.")
            if self.verbose:
                print(f"Hand winner: Player {winner} (+{self.current_pot})")
            self.points[winner] += self.current_pot
            self.last_hand_points = self.current_pot
            self.last_hand_winner = winner
            awarded_pot = self.current_pot
            self._record_event({
                "kind": "hand_end",
                "actor": winner,
                "winner": winner,
                "hand_points": awarded_pot,
            })
            self.current_pot = 0
            self._draw_phase(winner)
            self._start_new_hand(winner)
            return

        self.current_player = 1 - self.current_player

    # Compatibility method: play until hand resolves / game ends.
    def step(self):
        if self.done:
            return
        starting_hand_points = (self.points[0], self.points[1], self.last_hand_winner, self.current_pot, self.done)
        while not self.done:
            hand_signature = (self.points[0], self.points[1], self.last_hand_winner, self.current_pot)
            infoset = self.get_infoset(self.current_player)
            action = self.players[self.current_player].act(infoset)
            self.apply_action(action)
            if self.done:
                break
            new_signature = (self.points[0], self.points[1], self.last_hand_winner, self.current_pot)
            if new_signature != hand_signature and self.current_pot == 0:
                break
        return starting_hand_points

    def snapshot(self):
        return {
            "seed": self.seed,
            "deck_cards": deepcopy(self.deck.cards),
            "hands": deepcopy(self.hands),
            "points": deepcopy(self.points),
            "done": self.done,
            "played_cards": deepcopy(self.played_cards),
            "played_rank_counts": deepcopy(self.played_rank_counts),
            "last_hand_points": self.last_hand_points,
            "last_hand_winner": self.last_hand_winner,
            "face_up": deepcopy(self.face_up),
            "current_player": self.current_player,
            "hand_type": self.hand_type,
            "last_move": deepcopy(self.last_move),
            "last_player": self.last_player,
            "last_action_was_pass": self.last_action_was_pass,
            "pass_count": self.pass_count,
            "current_pot": self.current_pot,
            "public_history": deepcopy(list(self.public_history)),
            "history_maxlen": self.history_maxlen,
        }

    @classmethod
    def from_snapshot(cls, snapshot, players, verbose=False):
        env = cls(players, seed=snapshot.get("seed"), verbose=verbose, history_maxlen=snapshot.get("history_maxlen", HISTORY_MAXLEN))
        env.players = players
        env.deck.cards = deepcopy(snapshot["deck_cards"])
        env.hands = deepcopy(snapshot["hands"])
        env.points = deepcopy(snapshot["points"])
        env.done = snapshot["done"]
        env.played_cards = deepcopy(snapshot["played_cards"])
        env.played_rank_counts = deepcopy(snapshot["played_rank_counts"])
        env.last_hand_points = snapshot["last_hand_points"]
        env.last_hand_winner = snapshot["last_hand_winner"]
        env.face_up = deepcopy(snapshot["face_up"])
        env.current_player = snapshot["current_player"]
        env.hand_type = snapshot["hand_type"]
        env.last_move = deepcopy(snapshot["last_move"])
        env.last_player = snapshot["last_player"]
        env.last_action_was_pass = snapshot["last_action_was_pass"]
        env.pass_count = snapshot["pass_count"]
        env.current_pot = snapshot["current_pot"]
        env.public_history = deque(deepcopy(snapshot["public_history"]), maxlen=env.history_maxlen)
        return env


# ======================
# Players
# ======================

class RandomPlayer:
    def act(self, infoset):
        return random.choice(infoset["legal_actions"])


class HumanPlayer:
    wants_concrete_same_rank_choices = True

    def act(self, infoset):
        hand = infoset["hand"]
        legal = infoset["legal_actions"]

        print("\nYour hand:")
        for i, c in enumerate(hand):
            print(f"{i}: {c}")

        print("\nLegal moves:")
        for i, move in enumerate(legal):
            if move is None:
                print(f"{i}: PASS")
            else:
                print(f"{i}: {[str(c) for c in move]}")

        while True:
            try:
                choice = int(input("Choose move index: "))
                if 0 <= choice < len(legal):
                    return legal[choice]
            except Exception:
                pass
            print("Invalid choice.")


if __name__ == "__main__":
    print("=== Five-Ten-K Playtest ===")
    env = GameEnv([HumanPlayer(), RandomPlayer()], verbose=True)
    step = 0
    while not env.done:
        print(f"\n=== STEP {step} ===")
        print("Points:", env.points)
        print("Deck:", env.deck.size())
        print("Current player:", env.current_player)
        env.step()
        step += 1
    print("\nGame Over")
    print(env.points)
