"""
webgame.py — web-facing game controller built on V5's GameEnv API.

Uses env.apply_action() / env.get_infoset() throughout, so game logic
(drawing, endgame bonus, done detection) is owned entirely by game.py.
The EndgameSolverPlayer wrapper gives the bot optimal run-out play.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from game import Card, GameEnv, Move, RandomPlayer
from endgame import EndgameSolverPlayer, endgame_act_from_infoset
from policy_loader import LoadedPolicy

HISTORY_MAXLEN = 200   # enough for a full game's public events


def rank_text(rank: int) -> str:
    return {11: "J", 12: "Q", 13: "K", 14: "A", 17: "2", 20: "SJ", 30: "BJ"}.get(rank, str(rank))


def suit_symbol(suit: Optional[str]) -> str:
    return {"H": "♥", "D": "♦", "C": "♣", "S": "♠"}.get(suit or "", "")


def card_label(card: Card) -> str:
    if card.rank == 20: return "SJ"
    if card.rank == 30: return "BJ"
    return f"{rank_text(card.rank)}{suit_symbol(card.suit)}"


def move_text(move) -> str:
    if move is None:
        return "PASS"
    cards = move.cards if isinstance(move, Move) else move
    m     = move if isinstance(move, Move) else Move(move)
    return f"{m.type.upper()}  {'  '.join(card_label(c) for c in cards)}"


class _PolicyPlayer:
    """Thin wrapper so EndgameSolverPlayer can call .act(infoset)."""
    def __init__(self, policy: LoadedPolicy):
        self._policy = policy
    def act(self, infoset):
        return self._policy.choose_action(infoset)["move"]


@dataclass
class WebMatchController:
    policy: LoadedPolicy
    bot_first: bool = False
    seed: Optional[int] = None

    session_id: str        = field(init=False, default_factory=lambda: uuid.uuid4().hex)
    human_seat: int        = field(init=False, default=0)
    bot_seat: int          = field(init=False, default=1)
    env: Optional[GameEnv] = field(init=False, default=None)
    bot_player: object     = field(init=False, default=None)
    log: List[str]         = field(init=False, default_factory=list)
    human_wins: int        = field(init=False, default=0)
    bot_wins: int          = field(init=False, default=0)
    _game_counted: bool    = field(init=False, default=False)

    def __post_init__(self):
        self.bot_player = EndgameSolverPlayer(_PolicyPlayer(self.policy))
        self.reset(initial=True)

    # ------------------------------------------------------------------
    def reset(self, initial: bool = False):
        players = [RandomPlayer(), RandomPlayer()]  # placeholders; we call apply_action directly
        self.env = GameEnv(players, seed=self.seed, verbose=False,
                           history_maxlen=HISTORY_MAXLEN)
        starter = self.env.current_player
        if self.bot_first:
            self.bot_seat  = starter
            self.human_seat = 1 - starter
        else:
            self.human_seat = starter
            self.bot_seat  = 1 - starter
        self.log = ["Game started", "Bot leads" if self.bot_first else "You lead"]
        self._game_counted = False
        if not initial:
            self._autoplay_bot()

    # ------------------------------------------------------------------
    def _seat_name(self, seat: int) -> str:
        return "you" if seat == self.human_seat else "bot"

    def _is_human_turn(self) -> bool:
        return not self.env.done and self.env.current_player == self.human_seat

    def _is_bot_turn(self) -> bool:
        return not self.env.done and self.env.current_player == self.bot_seat

    # ------------------------------------------------------------------
    def _do_apply(self, action):
        """Apply one action and append to log."""
        p = self.env.current_player
        name = self._seat_name(p)
        if action is None:
            self.log.append(f"{name} PASS")
        else:
            mv = Move(action)
            pts = sum(getattr(__import__("game"), "POINT_VALUES", {}).get(c.rank, 0) for c in action)
            verb = "play" if name == "you" else "plays"
            self.log.append(f"{name} {verb} {move_text(action)} (+{pts})")
        self.env.apply_action(action)

    def _autoplay_bot(self, max_steps: int = 400):
        steps = 0
        while not self.env.done and self._is_bot_turn() and steps < max_steps:
            infoset = self.env.get_infoset(self.bot_seat)
            action  = self.bot_player.act(infoset)
            self._do_apply(action)
            steps += 1
        self._count_finished_game()

    def _count_finished_game(self):
        if not self.env.done or self._game_counted:
            return
        hp = self.env.points[self.human_seat]
        bp = self.env.points[self.bot_seat]
        if hp > bp:  self.human_wins += 1
        elif bp > hp: self.bot_wins  += 1
        self._game_counted = True

    # ------------------------------------------------------------------
    def human_play_by_index(self, action_index: int):
        if not self._is_human_turn():
            raise ValueError("Not the human's turn.")
        infoset = self.env.get_infoset(self.human_seat)
        legal   = infoset["legal_actions"]
        if action_index < 0 or action_index >= len(legal):
            raise IndexError("Action index out of range.")
        self._do_apply(legal[action_index])
        self._count_finished_game()

    def bot_play_if_needed(self):
        self._autoplay_bot()

    # ------------------------------------------------------------------
    def _serialize_card(self, card: Card) -> dict:
        return {
            "rank":         card.rank,
            "suit":         card.suit,
            "label":        card_label(card),
            "key":          f"{card.rank}:{card.suit or ''}",
            "color":        "red" if card.suit in {"H", "D"} else "black",
            "rank_label":   rank_text(card.rank),
            "suit_symbol":  suit_symbol(card.suit),
        }

    def _serialize_action(self, action, idx: int) -> dict:
        if action is None:
            return {"index": idx, "is_pass": True, "type": "pass",
                    "cards": [], "label": "PASS", "key": "PASS"}
        mv    = action if isinstance(action, Move) else Move(action)
        cards = mv.cards if isinstance(action, Move) else action
        return {
            "index":   idx,
            "is_pass": False,
            "type":    mv.type,
            "cards":   [self._serialize_card(c) for c in cards],
            "label":   move_text(action),
            "key":     "|".join(sorted(f"{c.rank}:{c.suit or ''}" for c in cards)),
        }

    def _result_text(self) -> str:
        if not self.env.done: return ""
        hp = self.env.points[self.human_seat]
        bp = self.env.points[self.bot_seat]
        if hp > bp: return "Final Result: You win"
        if bp > hp: return "Final Result: Bot wins"
        return "Final Result: Tie game"

    def state_payload(self) -> dict:
        env   = self.env
        human_hand = sorted(env.hands[self.human_seat], key=lambda c: (c.rank, c.suit or ""))

        if self._is_human_turn() and not env.done:
            infoset     = env.get_infoset(self.human_seat)
            legal        = infoset["legal_actions"]
        else:
            legal = []

        serialized_legal = [self._serialize_action(a, i) for i, a in enumerate(legal)]
        playable_keys    = sorted({
            card["key"]
            for act in serialized_legal if not act["is_pass"]
            for card in act["cards"]
        })

        last_move_ser = None
        if env.last_move is not None:
            last_move_ser = self._serialize_action(env.last_move, -1)

        return {
            "session_id":          self.session_id,
            "game_id":             self.session_id,
            "bot_first":           self.bot_first,
            "scores": {
                "you": env.points[self.human_seat],
                "bot": env.points[self.bot_seat],
                "pot": env.current_pot,
            },
            "wins": {
                "you": self.human_wins,
                "bot": self.bot_wins,
            },
            "turn":             "you" if self._is_human_turn() else ("bot" if self._is_bot_turn() else "game_over"),
            "pending_bot_turn": self._is_bot_turn() and not env.done,
            "hand_type":        env.hand_type or "open",
            "deck_size":        env.deck.size(),
            "opponent_card_count": len(env.hands[self.bot_seat]),
            "human_hand":       [self._serialize_card(c) for c in human_hand],
            "legal_actions":    serialized_legal,
            "playable_card_keys": playable_keys,
            "last_move":        last_move_ser,
            "result":           self._result_text(),
            "done":             env.done,
            "log":              self.log[-250:],
        }


class SessionStore:
    def __init__(self, policy: LoadedPolicy):
        self.policy   = policy
        self.sessions: Dict[str, WebMatchController] = {}

    def create(self, bot_first: bool = False, seed: Optional[int] = None) -> WebMatchController:
        ctrl = WebMatchController(policy=self.policy, bot_first=bot_first, seed=seed)
        ctrl.bot_play_if_needed()
        self.sessions[ctrl.session_id] = ctrl
        return ctrl

    def get(self, session_id: str) -> WebMatchController:
        if session_id not in self.sessions:
            raise KeyError("Unknown session id.")
        return self.sessions[session_id]
