import itertools
import random
from enum import Enum

import numpy as np

# Number of colorable type-slots per color (the colored-block stride). All Type members
# EXCEPT the King are colorable; the King is a colorless wild with its own dedicated slot
# (like Change Color / Super TAKI) and is never placed in the colored block. This stride is
# deliberately decoupled from len(Type) so adding the wild-only King (len(Type) -> 16) does
# not shift every colored card's slot.
TYPES_PER_COLOR = 15
# Size of a single card / deck vector (see card_to_vector / card_to_scalar):
#   indices 0-59  -> the 60 colored card slots
#   index   60    -> colorless Change Color
#   index   61    -> Super TAKI
#   index   62    -> colorless King
CARD_VECTOR_SIZE = 63
# Number of real play colors (RED, YELLOW, GREEN, BLUE); Color.NONE is encoded as all-zero.
NUM_PLAY_COLORS = 4
# Action scalar space (see action_to_scalar / scalar_to_action):
#   0-59 colored plays, 60 colorless Change Color, 61 Super TAKI, 62 King, 63 DRAW,
#   64 CLOSE_TAKI
ACTION_SIZE = 65
# OBSERVATION_SIZE is defined just below the State enum (it depends on len(State)).


class Color(Enum):
    """
    An Enum representing the color of the card.
    """
    NONE = 0
    RED = 1
    YELLOW = 2 
    GREEN = 3
    BLUE = 4

    def __str__(self):
        if self is Color.NONE:
            return ""
        elif self is Color.RED:
            return "red"
        elif self is Color.YELLOW:
            return "yellow"
        elif self is Color.GREEN:
            return "green"
        elif self is Color.BLUE:
            return "blue"


class Type(Enum):
    """
    An Enum representing the type of the card.
    """
    TAKI = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    STOP = 10
    CHDIR = 11
    PLUSTWO = 12
    PLUS = 13
    CHCOL = 14
    KING = 15

    def __str__(self):
        if self is Type.TAKI:
            return "taki"
        elif self is Type.STOP:
            return "stop"
        elif self is Type.CHDIR:
            return "change direction"
        elif self is Type.PLUSTWO:
            return "2+"
        elif self is Type.CHCOL:
            return "change color"
        elif self is Type.PLUS:
            return "+"
        elif self is Type.KING:
            return "king"
        else:
            return str(self.value)


class Card:
    def __init__(self, cardtype, color=Color.NONE):
        """

        :param color(Color): The color of the card
        :param cardtype(Type): The type of the card
        """
        self.color = color
        self.type = cardtype

    def __str__(self):
        if self.type is Type.TAKI and self.color is Color.NONE:
            return "super taki"
        return f"{self.color}{' ' if not self.color is Color.NONE else ''}{self.type}"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if not isinstance(other, Card):
            return NotImplemented
        return self.type == other.type and self.color == other.color

    def __hash__(self):
        return hash((self.type, self.color))


class Action(Enum):
    """
    An Enum representing the possible actions.
    """
    PLAY_CARD = 0
    DRAW = 1
    CLOSE_TAKI = 2

    def __str__(self):
        """
        The string representing the action.
        :return: a string
        """
        if self is Action.PLAY_CARD:
            return "play"
        elif self is Action.DRAW:
            return "draw"
        else:
            return "close taki"


class State(Enum):
    """
    An Enum representing the possible states.
    """
    NORMAL = 0
    DRAW_TWO = 1
    TAKI = 2
    SUPER_TAKI = 3
    FINISHED = 4
    PLUS = 5
    STOP = 6  # Internal State
    KING = 7  # King played: same player may put one more card (any color/type)


# Plain number cards (ONE..NINE); the round must OPEN on one of these.
NUMBER_TYPE_VALUES = frozenset(range(Type.ONE.value, Type.NINE.value + 1))
# Card types the game may END on: everything except PLUS. PLUS obliges the player to put one
# more card, which an empty hand cannot satisfy; every other card's effect lands on someone
# else and is coherent as a final card. Ending on a PLUS triggers the penalty draw.
FINISHING_TYPE_VALUES = frozenset(t.value for t in Type) - {Type.PLUS.value}
# Cards dealt to each player at the start of a round.
INITIAL_HAND_SIZE = 8
# Observation normalisation constants (see Game.observation). Count features are
# rescaled by a single fixed divisor so they stay on a comparable, bounded scale
# regardless of game progress (the absolute counts are preserved, just in new units).
CARD_COPIES_NORM = 4    # max copies of any single card (the change-color cards)
MAX_PLUS_TWO_STACK = 8  # number of +2 cards, i.e. the largest meaningful draw_num
# Opponent hand sizes shown in the observation, in turn order from the current player. Fixed
# so the vector stays constant-length across the supported 2..10 player counts (zero-padded
# when there are fewer opponents, truncated when there are more).
OPP_HAND_SLOTS = 3
# One-hot of the number of seats at the table, so one net can be trained and played across
# player counts and tell them apart explicitly. The count is *nearly* inferable from the block
# above already (a zero-padded opponent slot means "no such seat"), but not cleanly: a real
# opponent holding 0 cards has just won and reads as the same zero. More to the point, the count
# changes the GAME and not just the table size — STOP is a free extra turn at 2 seats, and CHDIR
# is a no-op at 2 / identical to STOP at 3 — so the policy has to condition on it rather than
# infer it. Slots cover 2, 3 and 4-or-more seats; the last saturates, matching how the opponent
# hand slots already truncate above 4. A 1-player game does not exist, so 2 is the low end.
NUM_PLAYER_SLOTS = 3
MIN_PLAYERS = 2
# A fresh deck builds to this many cards (see _setup_round): 14 colored types x 4 colors x 2
# + 2 Super TAKI + 4 Change Color + 2 King = 120. Normaliser for the deck-size feature.
TOTAL_DECK_CARDS = 120
# Per-type copy counts, used to normalise the unseen-count features into [0, 1].
TOTAL_PLUS_TWO = 8   # 4 colors x 2
TOTAL_KING = 2
TOTAL_CHCOL = 4
# Extra scalar features appended to the observation (see Game.observation):
#   turn direction(1) + opponent hand sizes in turn order(OPP_HAND_SLOTS) + deck size(1)
#   + unseen +2 / King / Change-Color counts(3) + player-count one-hot(NUM_PLAYER_SLOTS) = 11.
EXTRA_FEATURES = 1 + OPP_HAND_SLOTS + 1 + 3 + NUM_PLAYER_SLOTS
# observation() = hand(63) + state one-hot(len(State)) + draw_num(1)
#                 + open-TAKI-color one-hot(4) + shown_card(63) + extra features(11)
# The discard pile is deliberately NOT exposed as a histogram (only the shown top card is);
# the net gets a coarse card-count sense via the unseen +2/King/CHCOL features instead, rather
# than a full memory of everything that has been played.
OBSERVATION_SIZE = CARD_VECTOR_SIZE * 2 + len(State) + 1 + NUM_PLAY_COLORS + EXTRA_FEATURES


def action_to_scalar(action, card):
    """
    Converts an action to a scalar.
    :param action: the action to convert
    :param card: the card played if the action is PLAY_CARD
    :return: the scalar representing that (Action, Card) pair
    """
    if action is Action.PLAY_CARD:
        if card.color is not Color.NONE:
            return (card.color.value-1) * TYPES_PER_COLOR + card.type.value  # Play any colored card
        elif card.type is Type.CHCOL:
            # Scalar 60 = a colorless Change Color play. Unreachable in practice: valid_moves
            # always expands a playable CHCOL into its four colored choices (scalars 14/29/44/59),
            # so this branch never fires from real play and action 60 is never trained. It exists
            # only to mirror the card-vector layout (card_to_scalar maps an in-hand CHCOL here).
            return 60  # Play a colorless Change Color
        elif card.type is Type.KING:
            return 62  # Play the colorless King
        return 61  # Play SUPER TAKI
    elif action is Action.DRAW:
        return 63
    elif action is Action.CLOSE_TAKI:
        return 64


def scalar_to_action(scalar):
    """
    Converts a scalar to an action.
    :param scalar: the scalar to convert.
    :return: an (Action, Card) tuple.
    """
    if scalar < 60:
        cardtype = Type(scalar % TYPES_PER_COLOR)
        color = Color((scalar - cardtype.value) // TYPES_PER_COLOR + 1)
        return Action.PLAY_CARD, Card(cardtype, color)
    elif scalar == 60:
        return Action.PLAY_CARD, Card(Type.CHCOL)
    elif scalar == 61:
        return Action.PLAY_CARD, Card(Type.TAKI)
    elif scalar == 62:
        return Action.PLAY_CARD, Card(Type.KING)
    elif scalar == 63:
        return Action.DRAW, None
    else:
        return Action.CLOSE_TAKI, None


def card_to_scalar(card):
    """
    Converts the card to a scalar.
    Finds the index in the card vector.
    :param card: the card to convert
    :return: a scalar (int)
    """
    if card.color is not Color.NONE:
        return (card.color.value - 1) * TYPES_PER_COLOR + card.type.value
    elif card.type is Type.CHCOL:
        return 60
    elif card.type is Type.KING:
        return 62
    else:
        return 61


def card_to_vector(card, *cards):
    """
    Converts a card / deck to a vector.
    The vector has the size of the number of cards, and the value of each index is the amount of cards of that type and
    color.
    :param card: the card object.
    :param cards: additional objects to add to the vector.
    :return: the card / deck vector/
    """
    vec = np.zeros(CARD_VECTOR_SIZE, dtype=int)
    vec[card_to_scalar(card)] = 1
    if len(cards) > 0:
        for c in cards:
            vec += card_to_vector(c)
    return vec


def state_to_vector(state):
    """One-hot encoding of a State (length len(State))."""
    vec = np.zeros(len(State), dtype=int)
    vec[state.value] = 1
    return vec


def color_to_vector(color):
    """
    One-hot encoding of a play color (length NUM_PLAY_COLORS).
    Color.NONE (no open TAKI) is encoded as the all-zero vector.
    """
    vec = np.zeros(NUM_PLAY_COLORS, dtype=int)
    if color is not Color.NONE:
        vec[color.value - 1] = 1
    return vec


# --- Symmetry tables (used by the DQN replay augmentation, agents/dqn.py) -------------
# The dynamics have two independent relabeling symmetries. Under either, a legal game maps
# onto a legal game with identical dynamics and (hand-size-based) rewards, so a relabeled
# transition is a genuine transition rather than noise:
#   COLOR: the four play colors are interchangeable (24 permutations).
#   RANK:  the nine number cards ONE..NINE are interchangeable (9! = 362,880 permutations).
#          Matching is "same color or same type", the deck holds 2 copies of every rank in
#          every color, and the two rules that mention ranks at all are set-membership tests
#          (the round must OPEN on a number, NUMBER_TYPE_VALUES; it may END on a number or
#          the King, FINISHING_TYPE_VALUES). No rule branches on a *specific* rank, so any
#          global relabeling of the nine ranks preserves the dynamics exactly.
# The two commute — a color perm moves whole colored blocks; a rank perm permutes the same
# nine slots inside every block — so they compose into ~8.7M relabelings (see sym_tables).
#
# Both are expressed with one pair of tables, in one convention:
#   act forward map:  act_f[a] = what old action scalar a is called after the relabeling.
#   obs gather array: relabeled_obs = obs[obs_gather].
# Direction matters: aug[forward[i]] = orig[i]  <=>  aug[j] = orig[forward^-1[j]], so the
# observation table is the INVERSE of the forward map (built by inverting obs_f below).
COLOR_PERMS = tuple(itertools.permutations(range(1, NUM_PLAY_COLORS + 1)))
# Rank slots within one colored block: ONE..NINE sit at type-value offsets 1..9. TAKI (0)
# and STOP/CHDIR/PLUSTWO/PLUS/CHCOL (10..14) are not ranks and never move, nor do the three
# colorless slots (CHCOL 60, Super TAKI 61, King 62).
RANK_OFFSETS = np.array(sorted(NUMBER_TYPE_VALUES), dtype=np.intp)
NUM_RANKS = len(RANK_OFFSETS)                            # 9
# Offset of the open-TAKI color one-hot, and of the two card-vector blocks in the
# observation: hand, shown (the discard histogram is not part of the observation).
_TAKI_COLOR_OFF = CARD_VECTOR_SIZE + len(State) + 1      # 72
_CARD_BLOCK_OFFS = (0, _TAKI_COLOR_OFF + NUM_PLAY_COLORS)


def _tables_from_card_maps(card_f, color_f=None):
    """Turn forward permutation(s) of the 63 card slots into (obs gathers, act forwards).

    card_f is (n, CARD_VECTOR_SIZE); color_f, if given, is the matching (n, NUM_PLAY_COLORS)
    forward map of the open-TAKI color one-hot (rank relabelings leave it alone).
    """
    n = len(card_f)
    # Action slots 0-59 mirror the colored card slots; 60-64 (CHCOL / Super TAKI / King /
    # DRAW / CLOSE_TAKI) are colorless and rankless, so they are fixed under both symmetries.
    act_f = np.tile(np.arange(ACTION_SIZE, dtype=np.intp), (n, 1))
    act_f[:, :60] = card_f[:, :60]
    # Forward observation map, then inverted into a gather array.
    obs_f = np.tile(np.arange(OBSERVATION_SIZE, dtype=np.intp), (n, 1))
    for off in _CARD_BLOCK_OFFS:
        obs_f[:, off:off + CARD_VECTOR_SIZE] = off + card_f
    if color_f is not None:
        obs_f[:, _TAKI_COLOR_OFF:_TAKI_COLOR_OFF + NUM_PLAY_COLORS] = _TAKI_COLOR_OFF + color_f
    # Everything else — the state one-hot, draw_num, and the extra features (turn direction,
    # opponent hand sizes, deck size, unseen +2/King/CHCOL counts, player-count one-hot) — is
    # invariant under both relabelings and stays at its own index.
    gather = np.empty((n, OBSERVATION_SIZE), dtype=np.intp)
    np.put_along_axis(gather, obs_f, np.tile(np.arange(OBSERVATION_SIZE), (n, 1)), axis=1)
    return gather, act_f


def color_perm_tables(color_perms):
    """(obs gathers, act forwards) for color relabelings; rows of COLOR_PERMS."""
    color_perms = np.asarray(color_perms, dtype=np.intp)          # (n, 4), 1-based colors
    n = len(color_perms)
    card_f = np.tile(np.arange(CARD_VECTOR_SIZE, dtype=np.intp), (n, 1))
    for c in range(NUM_PLAY_COLORS):
        # Colored block c moves wholesale to block color_perms[:, c] - 1.
        new_base = (color_perms[:, c, None] - 1) * TYPES_PER_COLOR
        card_f[:, c * TYPES_PER_COLOR:(c + 1) * TYPES_PER_COLOR] = (
            new_base + np.arange(TYPES_PER_COLOR))
    return _tables_from_card_maps(card_f, color_f=color_perms - 1)


def rank_perm_tables(rank_perms):
    """(obs gathers, act forwards) for rank relabelings.

    rank_perms is (n, NUM_RANKS); row r is a permutation of range(NUM_RANKS) meaning the
    rank at block offset RANK_OFFSETS[i] is relabeled to RANK_OFFSETS[rank_perms[r, i]].
    The same permutation is applied inside every colored block — a rank relabeling is
    global, not per-color (a per-color one would not preserve same-type matching).
    """
    rank_perms = np.asarray(rank_perms, dtype=np.intp)            # (n, 9)
    n = len(rank_perms)
    card_f = np.tile(np.arange(CARD_VECTOR_SIZE, dtype=np.intp), (n, 1))
    new_offs = RANK_OFFSETS[rank_perms]                           # (n, 9)
    for c in range(NUM_PLAY_COLORS):
        base = c * TYPES_PER_COLOR
        card_f[:, base + RANK_OFFSETS] = base + new_offs
    return _tables_from_card_maps(card_f)


OBS_PERMS, ACT_PERMS = color_perm_tables(COLOR_PERMS)     # shapes (24, 147) and (24, 65)


def sym_tables(n, color_sym=True, rank_sym=False):
    """Draw n independent relabelings; return (obs gathers, act forwards), or (None, None).

    One relabeling per row, drawn uniformly from whichever symmetry groups are enabled (the
    identity is included in both, so a row may be a no-op). With both on, the two are
    composed into a single gather/forward pair — they commute, so the order is immaterial.
    """
    obs_g = act_f = None
    if color_sym:
        # The 24 color tables are precomputed; index them rather than rebuilding.
        ks = np.random.randint(len(OBS_PERMS), size=n)
        obs_g, act_f = OBS_PERMS[ks], ACT_PERMS[ks]
    if rank_sym:
        # 9! is far too large to precompute, so the rank tables are built per batch. Uniform
        # random permutations via argsort of uniform noise.
        r_obs, r_act = rank_perm_tables(np.argsort(np.random.rand(n, NUM_RANKS), axis=1))
        if obs_g is None:
            obs_g, act_f = r_obs, r_act
        else:
            # Composition, in the two conventions: obs[Gc][Gr] == obs[Gc[Gr]] for gathers,
            # and a -> rank_f[color_f[a]] for the forward action maps.
            obs_g = np.take_along_axis(obs_g, r_obs, axis=1)
            act_f = np.take_along_axis(r_act, act_f, axis=1)
    return obs_g, act_f


class Game:
    """
    The general Game class.
    Controls the flow of the game.
    """
    def __init__(self, agents, debug=False, seed=None):
        """
        Initialises the game
        :param agents: the agents playing
        :param debug: whether to print out information
        :param seed: the seed for the random actions
        """
        assert 1 < len(agents) < 11
        self.agents = agents
        self.debug = debug
        self.random = random.Random(seed)
        self.curr = 0
        self.dir = 1
        self.taki_color = Color.NONE  # the active color during an open (Super) TAKI
        self._setup_round()

    def _setup_round(self):
        """
        Builds, shuffles and deals a fresh deck for a new round.
        """
        self.state = State.NORMAL
        self.draw_num = 0
        self.deck = []
        self.discard = []
        # Public table-event log, one entry per action, reset each round. Records ONLY what
        # every player at the table can see (who played/drew what, and the active color at
        # the time) — no hidden information — so agents with human-like memory (e.g. the
        # heuristic's "he drew while red was on top" inference) can consume it without
        # widening the information set. Entries are plain-value tuples, never Card refs
        # (CHCOL cards mutate in place and the discard pile gets recycled):
        #   ('play',  seat, type_value, color_value, prev_state_value, active_color_value)
        #   ('draw',  seat, n_drawn,                 prev_state_value, active_color_value)
        #   ('close', seat, top_type_value, top_color_value)
        #   ('penalty_draw', seat, n_drawn)
        # where active_color_value is the color a play had to match when the action was taken.
        # The DQN observation/training paths ignore this entirely.
        self.history = []
        # Build each card as a DISTINCT object. `[Card(...)] * n` would alias one object
        # into n deck slots (in-place card mutation would then hit every alias); distinct
        # objects keep counts identical (4 CHCOL, 2 King, 2 per colored card, 2 Super TAKI)
        # while removing that hazard.
        for t in Type:
            if t == Type.CHCOL:
                self.deck.extend(Card(Type.CHCOL) for _ in range(4))
            elif t == Type.KING:
                # The King is a colorless wild (like CHCOL): 2 colorless copies, no colored.
                self.deck.extend(Card(Type.KING) for _ in range(2))
            else:
                for color in Color:
                    if color is not Color.NONE:
                        self.deck.extend(Card(t, color) for _ in range(2))
            if t == Type.TAKI:
                self.deck.extend(Card(t) for _ in range(2))
        self.random.shuffle(self.deck)
        self.discard.append(self.deck.pop())
        # House-rule simplification (official Taki flips the top card as-is): the game
        # must open on a plain number card. Re-draw the
        # starting card (returning it to the deck) until that holds, so no action
        # card's effect is silently dropped at the start of the round.
        while self.discard[-1].type.value not in NUMBER_TYPE_VALUES:
            self.deck.append(self.discard.pop())
            self.random.shuffle(self.deck)
            self.discard.append(self.deck.pop())
        self.hands = []
        for i in range(len(self.agents)):
            a = []
            for j in range(INITIAL_HAND_SIZE):
                a.append(self.deck.pop())
            self.hands.append(a)

    def reset(self, start_seat=0, agents=None):
        """
        Resets the game state for a fresh game.
        :param start_seat: the seat that opens the round (default 0 preserves eval/main/demo
            behaviour). Randomised per trial in training so the learner no longer opens 100%
            of games (see A4).
        :param agents: optionally reseat the table for this round. Mixed-count training samples
            the seat count per trial, and a round is the only point where changing it is
            meaningful. Reusing the Game object (rather than constructing a new one per trial)
            is deliberate: the deck RNG lives on the instance, so a fresh Game per trial would
            restart the stream and deal every trial the SAME cards.
        """
        if agents is not None:
            assert len(agents) >= MIN_PLAYERS
            self.agents = agents
        assert 0 <= start_seat < len(self.agents)
        self.curr = start_seat
        self.dir = 1
        self.taki_color = Color.NONE
        self._setup_round()

    def shown_card(self):
        """
        Returns the card on top of the discard pile.
        :return: the top card
        """
        return self.discard[-1]

    def next_agent(self):
        """
        Calculates the next player
        :return: the next player
        """
        self.curr = (self.curr + self.dir) % len(self.agents)

    def draw_card(self, agent, amount=1):
        """
        Draws one or more cards for an agent
        :param agent: the agent to draw cards to
        :param amount: the amount of cards to draw
        """
        for i in range(amount):
            if len(self.deck) == 0:
                # Recycle the discard pile (except the shown card) back into the deck.
                if len(self.discard) <= 1:
                    # Nothing left to recycle (all cards are in players' hands); the
                    # deck is genuinely exhausted, so stop drawing rather than crash.
                    return
                self.deck.extend(self.discard[:-1])
                self.random.shuffle(self.deck)
                for card in self.deck:
                    if card.type is Type.CHCOL:  # Needs reset on change color
                        card.color = Color.NONE
                self.discard = self.discard[-1:]
            self.hands[agent].append(self.deck.pop())

    def process_action(self, action, card, agent):
        """
        Process an action
        :param action: the action to process
        :param card: the card played if action is PLAY_CARD
        :param agent: the agent who made the action
        """
        if self.debug:
            print(f"Player {agent+1}'s turn.\nCards {self.hands[agent]}")
        # The state we entered this action with, before the transient resets below. Used to
        # tell "closing an open TAKI" from "declining a King's optional follow-up" (both use
        # CLOSE_TAKI), since the KING reset erases self.state.
        prev_state = self.state
        # The color a play had to match at the moment of this action (for the history log):
        # the open TAKI's color if one is running, else the shown card's color. Captured
        # before the action mutates the pile/state.
        if prev_state is State.TAKI or prev_state is State.SUPER_TAKI:
            active_color = self.taki_color
        else:
            active_color = self.shown_card().color
        if self.state is State.PLUS:
            self.state = State.NORMAL
        if self.state is State.SUPER_TAKI:
            self.state = State.TAKI
        if self.state is State.KING:
            # The King's follow-up card (or a CLOSE_TAKI decline) consumes the King
            # continuation; process it in a normal context so its own effect applies and a
            # plain number follow-up ends the turn.
            self.state = State.NORMAL
        if action == Action.PLAY_CARD:
            self.history.append(
                ('play', agent, card.type.value, card.color.value,
                 prev_state.value, active_color.value))
            self.discard.append(card)
            if card.type is Type.CHCOL:
                self.hands[agent].remove(Card(Type.CHCOL))
            else:
                if card not in self.hands[agent]:
                    raise ValueError(
                        f"agent {agent} played {card}, which is not in hand {self.hands[agent]}")
                self.hands[agent].remove(card)
            if card.type is Type.TAKI:
                if card.color is Color.NONE:
                    self.state = State.SUPER_TAKI
                    # A Super TAKI acts as a TAKI of the color of the card beneath it.
                    self.taki_color = self.discard[-2].color if len(self.discard) >= 2 else Color.NONE
                else:
                    self.state = State.TAKI
                    self.taki_color = card.color
            elif card.type is Type.PLUSTWO:
                if self.state is not State.TAKI and self.state is not State.SUPER_TAKI:
                    self.state = State.DRAW_TWO
                    self.draw_num += 1
            elif card.type is Type.CHCOL:
                pass  # it automatically changes the color of the card
            elif card.type is Type.STOP:
                if self.state is not State.TAKI and self.state is not State.SUPER_TAKI:
                    self.state = State.STOP
            elif card.type is Type.CHDIR:
                # Inside a TAKI only the last card's effect applies; defer to CLOSE_TAKI.
                if self.state is not State.TAKI and self.state is not State.SUPER_TAKI:
                    self.dir *= -1
            elif card.type is Type.PLUS:
                if self.state is not State.TAKI and self.state is not State.SUPER_TAKI:
                    self.state = State.PLUS
            elif card.type is Type.KING:
                # Inside a TAKI only the last card's effect applies, so a King played
                # mid-sequence is inert (no +2 cancel, no color change, TAKI continues); its
                # "grant a follow-up" fires at CLOSE_TAKI if it is the card closed on. Outside
                # a TAKI the King is a colorless wild that cancels any pending +2
                # (draw_num -> 0) and keeps the turn for one optional follow-up card.
                if self.state is not State.TAKI and self.state is not State.SUPER_TAKI:
                    self.draw_num = 0
                    self.taki_color = Color.NONE
                    self.state = State.KING
            if self.debug:
                print(f"Player {agent+1} played {str(card)}.")
        elif action is Action.CLOSE_TAKI:
            self.history.append(
                ('close', agent, self.shown_card().type.value, self.shown_card().color.value))
            self.taki_color = Color.NONE
            if prev_state is State.KING:
                # Declining the King's optional follow-up: just end the turn.
                self.state = State.NORMAL
            elif self.shown_card().type is Type.STOP:
                self.state = State.STOP
            elif self.shown_card().type is Type.PLUS:
                self.state = State.PLUS
            elif self.shown_card().type is Type.PLUSTWO:
                self.state = State.DRAW_TWO
                self.draw_num += 1
            elif self.shown_card().type is Type.CHDIR:
                self.dir *= -1
                self.state = State.NORMAL
            elif self.shown_card().type is Type.KING:
                # The TAKI closed on a King: grant the optional follow-up turn.
                self.state = State.KING
            else:
                self.state = State.NORMAL
            if self.debug:
                print(f"Player {agent+1} closed the TAKI.")
        elif action is Action.DRAW:
            s = 0
            if self.state is State.DRAW_TWO:
                s += 2 * self.draw_num
                self.draw_num = 0
                self.state = State.NORMAL
            if self.state is State.TAKI or self.state is State.SUPER_TAKI:
                self.state = State.NORMAL
            if s == 0:
                s = 1
            self.history.append(('draw', agent, s, prev_state.value, active_color.value))
            self.draw_card(agent, s)
            if self.debug:
                print(f"Player {agent+1} drew {s} cards.")

    def valid_moves(self, agent=None):
        """
        Returns an array with valid moves for the given agent.
        :param agent: the agent to get moves of, defaults to the current player
        :return: an array of (Action, Card) tuples.
        """
        if agent is None:
            agent = self.curr
        cards = self.hands[agent]
        res = []
        in_taki = self.state is State.TAKI or self.state is State.SUPER_TAKI
        # Play Cards
        for card in cards:
            if self.state is State.DRAW_TWO:
                # Only another +2 (to stack the draw) or a King (to cancel it) may be
                # played in response to a pending +2; otherwise draw.
                if card.type is Type.PLUSTWO or card.type is Type.KING:
                    res.append((Action.PLAY_CARD, card))
            elif self.state is State.KING:
                # After a King the player may put any single card (any color/type);
                # wilds expand into their colored choices. Another King is allowed too
                # (Kings chain, each granting one more card).
                if card.type is Type.CHCOL:
                    for i in Color:
                        if i is not Color.NONE:
                            res.append((Action.PLAY_CARD, Card(Type.CHCOL, i)))
                else:
                    res.append((Action.PLAY_CARD, card))
            elif in_taki:
                # During an open (Super) TAKI only cards of the TAKI's color may be
                # played, plus the colorless wilds (incl. the King, which is inert mid-run
                # but grants a follow-up if the TAKI is closed on it). taki_color NONE means
                # "any" (fallback).
                if self.taki_color is Color.NONE \
                        or card.color is self.taki_color \
                        or card.color is Color.NONE:
                    if card.type is Type.CHCOL:
                        for i in Color:
                            if i is not Color.NONE:
                                res.append((Action.PLAY_CARD, Card(Type.CHCOL, i)))
                    else:
                        res.append((Action.PLAY_CARD, card))
            elif self.shown_card().type is card.type \
                    or self.shown_card().color is card.color \
                    or self.shown_card().color is Color.NONE\
                    or card.color is Color.NONE:
                if card.type is Type.CHCOL:
                    for i in Color:
                        if i is not Color.NONE:
                            res.append((Action.PLAY_CARD, Card(Type.CHCOL, i)))
                else:
                    res.append((Action.PLAY_CARD, card))
        # Draw Cards (not while an open TAKI or a King continuation is in progress — those
        # are ended by closing them, not by drawing)
        if not in_taki and self.state is not State.KING:
            res.append((Action.DRAW, None))
        # Close TAKI — also the "done, decline the follow-up" terminator for a King
        if in_taki or self.state is State.KING:
            res.append((Action.CLOSE_TAKI, None))
        # Collapse exact-duplicate moves (e.g. two identical cards in hand -> one entry;
        # a CHCOL's four colored choices are distinct and survive). Without this, a uniform
        # chooser (RandomAgent, epsilon-exploration) over-weights duplicated cards. Greedy
        # argmax play is unaffected (duplicates share a Q-value), so model-vs-model evals
        # are unchanged; only the random baseline's move distribution shifts.
        return list(dict.fromkeys(res))

    def next_turn(self):
        """
        Calculates next turn.
        """
        if self.done():
            return True, self.curr
        agent = self.agents[self.curr]
        if self.debug:
            print(f"It's player {self.curr+1}'s turn.")
        action, card = agent.play(self)
        self.process_action(action, card, self.curr)
        finished = len(self.hands[self.curr]) == 0
        if finished and action is Action.PLAY_CARD \
                and card.type.value not in FINISHING_TYPE_VALUES:
            # You may not end the game on a PLUS — it obliges you to play another card
            # (see FINISHING_TYPE_VALUES). Draw a penalty card and keep playing (the card's
            # own effect still applies).
            self.history.append(('penalty_draw', self.curr, 1))
            self.draw_card(self.curr, 1)
            # Only truly finished if the deck was exhausted and no card could be drawn.
            finished = len(self.hands[self.curr]) == 0
            if self.debug and not finished:
                print(f"Player {self.curr+1} can't finish on a PLUS; drew a penalty.")
        if finished:
            self.state = State.FINISHED
            if self.debug:
                print(f"Player {self.curr+1} won!")
            return True, self.curr
        if self.state == State.STOP:
            self.next_agent()
            self.state = State.NORMAL
        if self.state == State.NORMAL or self.state == State.DRAW_TWO:
            self.next_agent()
        return False, self.curr

    def done(self):
        """
        Returns whether the game is finished or not.
        :return: whether the game is finished or not.
        """
        return self.state == State.FINISHED

    def observation(self, agent=None):
        """
        Returns the state vector.
        :param agent: the agent to see the observation with.
        :return: the observation vector.
        """
        # hand + state(one-hot) + draw_num + open-TAKI-color(one-hot) + card shown
        #      + [direction, OPP_HAND_SLOTS opponent hand sizes in turn order, deck size,
        #         unseen +2 / King / Change-Color counts, player-count one-hot]
        # The discard pile is intentionally NOT exposed as a histogram; only the shown top
        # card is, plus the coarse unseen +2/King/CHCOL counts (see A7 scoping).
        # Count features are normalised (see the *_NORM / TOTAL_* constants) so every input
        # sits on a comparable, bounded scale; one-hot blocks and direction are left as-is.
        if agent is None:
            agent = self.curr
        num_players = len(self.agents)
        hand = self.hands[agent]
        hand_vec = (card_to_vector(*hand) if len(hand) > 0
                    else np.zeros(CARD_VECTOR_SIZE, dtype=int))
        # Opponent hand sizes in TURN ORDER from the current player (dir-aware), normalised by
        # INITIAL_HAND_SIZE. Fixed OPP_HAND_SLOTS slots: zero-padded when there are fewer
        # opponents, truncated when there are more, so the vector stays constant-length.
        opp_sizes = []
        for k in range(1, OPP_HAND_SLOTS + 1):
            if k < num_players:
                seat = (agent + k * self.dir) % num_players
                opp_sizes.append(len(self.hands[seat]) / INITIAL_HAND_SIZE)
            else:
                opp_sizes.append(0.0)
        # Unseen counts of +2 / King / Change-Color, counted by card TYPE over the raw Card
        # lists (a played CHCOL is recoloured into a colored slot, so slot-based counting is
        # ambiguous). Unseen = total - (in this agent's hand) - (in the discard pile).
        def _unseen(ctype, total):
            seen = (sum(1 for c in hand if c.type is ctype)
                    + sum(1 for c in self.discard if c.type is ctype))
            return (total - seen) / total
        # Player-count one-hot over 2 / 3 / 4-or-more seats; saturating at the top like the
        # opponent hand slots, which cannot show a 5th seat either.
        n_players_vec = np.zeros(NUM_PLAYER_SLOTS)
        n_players_vec[min(num_players, MIN_PLAYERS + NUM_PLAYER_SLOTS - 1) - MIN_PLAYERS] = 1
        extra = np.array([self.dir,
                          *opp_sizes,
                          len(self.deck) / TOTAL_DECK_CARDS,
                          _unseen(Type.PLUSTWO, TOTAL_PLUS_TWO),
                          _unseen(Type.KING, TOTAL_KING),
                          _unseen(Type.CHCOL, TOTAL_CHCOL),
                          *n_players_vec])
        return np.concatenate(
            (hand_vec / CARD_COPIES_NORM,
             state_to_vector(self.state),
             np.array([self.draw_num / MAX_PLUS_TWO_STACK]),
             color_to_vector(self.taki_color),
             card_to_vector(self.shown_card()),
             extra))
