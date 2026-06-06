"""
Information-Set MCTS (IS-MCTS) for Five-Ten-K inference.

The problem with vanilla PIMC (what SearchModelPlayer was doing):
  - It evaluates each *action* independently over N determinizations.
  - That means move A can look good because it wins in world W1, and move B
    because it wins in world W2, but you can only pick one move for all worlds.
  - This is the "strategy-fusion" bias: PIMC systematically picks bluffs and
    overplays that only work in one possible opponent hand.

IS-MCTS fixes this by building a SINGLE tree rooted at the *information set*
(what both players publicly know), sampling a determinization PER SIMULATION,
and accumulating visit counts and values in the same nodes regardless of which
determinization was used. Selection uses UCB (we always know the root player's
cards exactly; uncertainty is in the opponent's hand). The resulting policy is
much less exploitable and handles the "will they punish my pair-10s?" case
correctly.

For endgame positions (deck empty) we call the exact minimax solver instead of
rolling out -- this gives exact values with zero noise.

Design:
  - Nodes keyed on the PUBLIC game state (move sequence, points, deck progress)
    not the hidden state.
  - Opponent moves are sampled from a belief-weighted distribution of their hand
    (using the neural net's belief head for priors).
  - Rollout policy = neural network policy (fast, informed).
  - Endgame cutoff = exact solver (deck empty).
  - Tree is rebuilt each decision (no persistence across decisions -- correct for
    imperfect information where tree reuse would leak hidden state).
"""
import math
import random
import copy
from collections import defaultdict

import numpy as np
import torch

from game import GameEnv, Move, RANKS, HISTORY_MAXLEN
from endgame import endgame_value_from_env, endgame_act_from_infoset, ALL_CARDS
from encoder import encode_state, encode_move
from model import HISTORY_EVENT_DIM, encode_history_events

# Value-head output is in (margin / POINT_SCALE) units; multiply to recover margin.
# MUST match train_ppo.POINT_SCALE so the search leaves are on the same scale.
POINT_SCALE = 10.0

UCB_C = 1.5       # exploration constant
MAX_SIMS = 80     # simulations per decision (raise for stronger play / more time)
ROLLOUT_DEPTH = 6 # max rollout depth past the tree (capped by value net bootstrap)
MIN_SIMS_EXPAND = 2  # expand a node after this many visits


class ISNode:
    __slots__ = ("visits", "total", "children", "prior")

    def __init__(self, prior=1.0):
        self.visits = 0
        self.total = 0.0
        self.children = {}   # action_idx -> ISNode
        self.prior = prior

    def value(self):
        return self.total / self.visits if self.visits else 0.0

    def ucb(self, parent_visits, c=UCB_C):
        exploit = self.value()
        explore = c * self.prior * math.sqrt(math.log(parent_visits + 1) / (self.visits + 1))
        return exploit + explore


class ISMCTSPlayer:
    """IS-MCTS over the policy net's action priors and value estimates,
    with exact endgame resolution and belief-biased opponent sampling."""

    wants_concrete_same_rank_choices = False

    def __init__(self, model, device, simulations=MAX_SIMS, rollout_depth=ROLLOUT_DEPTH):
        self.model = model
        self.device = device
        self.sims = simulations
        self.depth = rollout_depth
        self._memo = {}   # endgame value cache (cleared between decisions)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def act(self, infoset):
        legal = infoset["legal_actions"]
        if len(legal) == 1:
            return legal[0]

        # Exact endgame: delegate immediately
        if infoset.get("deck_size", 1) == 0:
            return endgame_act_from_infoset(infoset)

        me = infoset["player_index"]
        priors, value_est = self._policy_priors(infoset)
        belief = self._opp_belief(infoset)

        root = ISNode()
        root.visits = 1
        for i, p in enumerate(priors):
            root.children[i] = ISNode(prior=float(p))

        self._memo = {}
        for _ in range(self.sims):
            # Fresh determinization each simulation -- IS-MCTS correctness
            snap = self._determinize(infoset, me, belief)
            env = GameEnv.from_snapshot(snap, [_NullPlayer(), _NullPlayer()], verbose=False)
            self._simulate(root, env, me, infoset, depth=0)

        # Pick by visit count (exploitation, not UCB)
        best_i = max(root.children, key=lambda i: root.children[i].visits)
        return legal[best_i]

    # ------------------------------------------------------------------
    # Tree simulation
    # ------------------------------------------------------------------

    def _simulate(self, node, env, me, root_infoset, depth):
        if env.done:
            margin = env.points[me] - env.points[1 - me]
            node.visits += 1
            node.total += margin
            return margin

        cur = env.current_player

        # Exact endgame
        if env.deck.size() == 0:
            key = self._env_key(env)
            if key not in self._memo:
                self._memo[key] = endgame_value_from_env(env, me)
            val = self._memo[key]
            node.visits += 1
            node.total += val
            return val

        legal = env.get_legal_actions(cur)

        if cur == me:
            # Our decision node -- use tree policy
            if not node.children or depth == 0 and node.visits < MIN_SIMS_EXPAND:
                val = self._rollout(env, me, depth)
                node.visits += 1
                node.total += val
                return val

            best_i = max(node.children, key=lambda i: node.children[i].ucb(node.visits))
            action = legal[best_i]
            child = node.children[best_i]

            env2 = self._step(env, action)
            val = self._simulate(child, env2, me, root_infoset, depth + 1)
        else:
            # Opponent's decision -- sample from policy prior (no tree for them)
            action = self._sample_opp_action(env, cur, legal)
            env2 = self._step(env, action)
            val = self._simulate(node, env2, me, root_infoset, depth)

        node.visits += 1
        node.total += val
        return val

    # ------------------------------------------------------------------
    # Rollout / evaluation
    # ------------------------------------------------------------------

    def _rollout(self, env, me, depth_so_far):
        """Short rollout then bootstrap with the value net."""
        env = copy.deepcopy(env)
        steps = 0
        while not env.done and steps < self.depth:
            if env.deck.size() == 0:
                key = self._env_key(env)
                if key not in self._memo:
                    self._memo[key] = endgame_value_from_env(env, me)
                return self._memo[key]
            cur = env.current_player
            info = env.get_infoset(cur)
            # Use the model's policy for rollout (informed, fast)
            act = self._net_act(info, greedy=False)
            env.apply_action(act)
            steps += 1
        if env.done:
            return env.points[me] - env.points[1 - me]
        # Bootstrap with the value head (already in margin/POINT_SCALE units).
        info = env.get_infoset(me)
        _, val = self._policy_priors(info)
        return POINT_SCALE * val

    def _net_act(self, infoset, greedy=True):
        """Sample (or pick best) action from the neural net policy."""
        legal = infoset["legal_actions"]
        state = encode_state(infoset)
        hist = encode_history_events(infoset)
        afeat = np.stack([encode_move(a, infoset) for a in legal]).astype(np.float32)
        with torch.no_grad():
            st = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            hlen = len(hist)
            ht = torch.from_numpy(hist if hlen else np.zeros((0, HISTORY_EVENT_DIM), dtype=np.float32)).float().unsqueeze(0).to(self.device)
            hl = torch.tensor([hlen], dtype=torch.long, device=self.device)
            at = torch.from_numpy(afeat).float().unsqueeze(0).to(self.device)
            logits, _, _ = self.model.score_actions(st, ht, hl, at)
        logits = logits[0].cpu()
        if greedy:
            idx = int(torch.argmax(logits).item())
        else:
            idx = int(torch.distributions.Categorical(logits=logits).sample().item())
        return legal[idx]

    def _sample_opp_action(self, env, cur, legal):
        """Sample opponent action from net policy (used in simulation rollout)."""
        info = env.get_infoset(cur)
        return self._net_act(info, greedy=False)

    # ------------------------------------------------------------------
    # Belief / determinization
    # ------------------------------------------------------------------

    def _opp_belief(self, infoset):
        """Per-rank probability estimate for the opponent's hand from the belief head."""
        state = encode_state(infoset)
        hist = encode_history_events(infoset)
        with torch.no_grad():
            st = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            hlen = len(hist)
            ht = torch.from_numpy(hist if hlen else np.zeros((0, HISTORY_EVENT_DIM), dtype=np.float32)).float().unsqueeze(0).to(self.device)
            hl = torch.tensor([hlen], dtype=torch.long, device=self.device)
            _, aux = self.model.compute_context(st, ht, hl)
            probs = torch.sigmoid(aux["opp_rank"])[0].cpu().numpy()
        return {RANKS[i]: float(probs[i]) for i in range(len(RANKS))}

    def _policy_priors(self, infoset):
        """Softmax policy priors and value for all legal actions."""
        legal = infoset["legal_actions"]
        state = encode_state(infoset)
        hist = encode_history_events(infoset)
        afeat = np.stack([encode_move(a, infoset) for a in legal]).astype(np.float32)
        with torch.no_grad():
            st = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            hlen = len(hist)
            ht = torch.from_numpy(hist if hlen else np.zeros((0, HISTORY_EVENT_DIM), dtype=np.float32)).float().unsqueeze(0).to(self.device)
            hl = torch.tensor([hlen], dtype=torch.long, device=self.device)
            at = torch.from_numpy(afeat).float().unsqueeze(0).to(self.device)
            logits, values, _ = self.model.score_actions(st, ht, hl, at)
        priors = torch.softmax(logits[0], dim=-1).cpu().numpy()
        value = float(values[0].item())
        return priors, value

    def _determinize(self, infoset, me, belief, floor=0.15):
        """Sample a determinization of the hidden state (opponent's hand + deck)
        weighted by the belief head's per-rank estimates."""
        opp = 1 - me
        seen = set((c.rank, c.suit or "") for c in infoset["hand"])
        for c in infoset["played_cards"]:
            seen.add((c.rank, c.suit or ""))
        unseen = [c for c in ALL_CARDS if (c.rank, c.suit or "") not in seen]
        n_opp = min(infoset.get("opp_card_count", 0), len(unseen))
        if n_opp == 0 or n_opp == len(unseen):
            random.shuffle(unseen)
            opp_hand, deck = unseen[:n_opp], unseen[n_opp:]
        else:
            w = np.array([belief.get(c.rank, 0.25) + floor for c in unseen], dtype=np.float64)
            w /= w.sum()
            idx = np.random.choice(len(unseen), size=n_opp, replace=False, p=w)
            idx_set = set(idx.tolist())
            opp_hand = [unseen[i] for i in idx]
            deck = [unseen[i] for i in range(len(unseen)) if i not in idx_set]
        played = list(infoset["played_cards"])
        prc = {r: 0 for r in RANKS}
        for c in played: prc[c.rank] += 1
        return {
            "seed": None, "deck_cards": deck,
            "hands": {me: list(infoset["hand"]), opp: opp_hand},
            "points": dict(infoset["points"]), "done": False,
            "played_cards": played, "played_rank_counts": prc,
            "last_hand_points": infoset.get("last_hand_points", 0),
            "last_hand_winner": infoset.get("last_hand_winner"),
            "face_up": {}, "current_player": me,
            "hand_type": infoset.get("hand_type"),
            "last_move": infoset.get("last_move"),
            "last_player": infoset.get("last_player"),
            "last_action_was_pass": infoset.get("last_action_was_pass", False),
            "pass_count": infoset.get("pass_count", 0),
            "current_pot": infoset.get("current_pot", 0),
            "public_history": list(infoset.get("public_history", [])),
            "history_maxlen": HISTORY_MAXLEN,
        }

    @staticmethod
    def _step(env, action):
        env2 = copy.deepcopy(env)
        env2.apply_action(action)
        return env2

    @staticmethod
    def _env_key(env):
        return (
            tuple(sorted((c.rank, c.suit or "") for c in env.hands[0])),
            tuple(sorted((c.rank, c.suit or "") for c in env.hands[1])),
            env.current_player, env.hand_type, env.current_pot,
            env.last_player,
        )


class _NullPlayer:
    """Placeholder -- actual decisions made by the MCTS simulation, not this."""
    def act(self, infoset):
        return infoset["legal_actions"][0]
