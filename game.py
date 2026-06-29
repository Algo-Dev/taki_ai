import random
from enum import Enum

import numpy as np

# Size of a single card / deck vector (see card_to_vector / card_to_scalar):
#   indices 0-59  -> the 60 colored card slots
#   index   60    -> colorless Change Color
#   index   61    -> Super TAKI
CARD_VECTOR_SIZE = 62
# Number of real play colors (RED, YELLOW, GREEN, BLUE); Color.NONE is encoded as all-zero.
NUM_PLAY_COLORS = 4
# Action scalar space (see action_to_scalar / scalar_to_action):
#   0-59 colored plays, 60 colorless Change Color, 61 Super TAKI, 62 DRAW, 63 CLOSE_TAKI
ACTION_SIZE = 64
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

    def amount(self):
        """
        Calculates the amount needed per game
        :return: an int
        """
        if self.type == Type.CHCOL:
            return 4
        return 2

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


# Number of plain number cards (ONE..NINE); only these may end the game.
NUMBER_TYPE_VALUES = frozenset(range(Type.ONE.value, Type.NINE.value + 1))
# Extra scalar features appended to the observation (see Game.observation):
#   turn direction, next player's hand size, minimum opponent hand size.
EXTRA_FEATURES = 3
# observation() = hand(62) + discard(62) + state one-hot(len(State)) + draw_num(1)
#                 + open-TAKI-color one-hot(4) + shown_card(62) + extra features(3)
OBSERVATION_SIZE = CARD_VECTOR_SIZE * 3 + len(State) + 1 + NUM_PLAY_COLORS + EXTRA_FEATURES


def action_to_scalar(action, card):
    """
    Converts an action to a scalar.
    :param action: the action to convert
    :param card: the card played if the action is PLAY_CARD
    :return: the scalar representing that (Action, Card) pair
    """
    if action is Action.PLAY_CARD:
        if card.color is not Color.NONE:
            return (card.color.value-1) * 15 + card.type.value  # Play any colored card
        elif card.type is Type.CHCOL:
            return 60  # Play a colorless Change Color
        return 61  # Play SUPER TAKI
    elif action is Action.DRAW:
        return 62
    elif action is Action.CLOSE_TAKI:
        return 63


def scalar_to_action(scalar):
    """
    Converts a scalar to an action.
    :param scalar: the scalar to convert.
    :return: an (Action, Card) tuple.
    """
    if scalar < 60:
        cardtype = Type(scalar % 15)
        color = Color((scalar - cardtype.value) // 15 + 1)
        return Action.PLAY_CARD, Card(cardtype, color)
    elif scalar == 60:
        return Action.PLAY_CARD, Card(Type.CHCOL)
    elif scalar == 61:
        return Action.PLAY_CARD, Card(Type.TAKI)
    elif scalar == 62:
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
        return (card.color.value - 1) * 15 + card.type.value
    elif card.type is Type.CHCOL:
        return 60
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
        for t in Type:
            if t == Type.CHCOL:
                self.deck.extend([Card(Type.CHCOL)] * 4)
            else:
                for color in Color:
                    if color is not Color.NONE:
                        self.deck.extend([Card(t, color)] * 2)
            if t == Type.TAKI:
                self.deck.extend([Card(t)] * 2)
        self.random.shuffle(self.deck)
        self.discard.append(self.deck.pop())
        # Standard Taki: the game must open on a plain number card. Re-draw the
        # starting card (returning it to the deck) until that holds, so no action
        # card's effect is silently dropped at the start of the round.
        while self.discard[-1].type.value not in NUMBER_TYPE_VALUES:
            self.deck.append(self.discard.pop())
            self.random.shuffle(self.deck)
            self.discard.append(self.deck.pop())
        self.hands = []
        for i in range(len(self.agents)):
            a = []
            for j in range(8):
                a.append(self.deck.pop())
            self.hands.append(a)

    def reset(self):
        """
        Resets the game state for a fresh game.
        """
        self.curr = 0
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
        if self.state is State.PLUS:
            self.state = State.NORMAL
        if self.state is State.SUPER_TAKI:
            self.state = State.TAKI
        if action == Action.PLAY_CARD:
            self.discard.append(card)
            if card.type is Type.CHCOL:
                self.hands[agent].remove(Card(Type.CHCOL))
            else:
                if card not in self.hands[agent]:
                    print(self.hands[agent], card)
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
            if self.debug:
                print(f"Player {agent+1} played {str(card)}.")
        elif action is Action.CLOSE_TAKI:
            self.taki_color = Color.NONE
            if self.shown_card().type is Type.STOP:
                self.state = State.STOP
            elif self.shown_card().type is Type.PLUS:
                self.state = State.PLUS
            elif self.shown_card().type is Type.PLUSTWO:
                self.state = State.DRAW_TWO
                self.draw_num += 1
            elif self.shown_card().type is Type.CHDIR:
                self.dir *= -1
                self.state = State.NORMAL
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
                # Only another +2 may be played to stack onto the draw; otherwise draw.
                if card.type is Type.PLUSTWO:
                    res.append((Action.PLAY_CARD, card))
            elif in_taki:
                # During an open (Super) TAKI only cards of the TAKI's color may be
                # played, plus the colorless wilds. taki_color NONE means "any" (fallback).
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
        # Draw Cards (not while an open TAKI is in progress — it is ended by closing it)
        if not in_taki:
            res.append((Action.DRAW, None))
        # Close TAKI
        if in_taki:
            res.append((Action.CLOSE_TAKI, None))
        return res

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
                and card.type.value not in NUMBER_TYPE_VALUES:
            # Standard Taki: you may not end the game on an action card. Draw a
            # penalty card and keep playing (the card's own effect still applies).
            self.draw_card(self.curr, 1)
            # Only truly finished if the deck was exhausted and no card could be drawn.
            finished = len(self.hands[self.curr]) == 0
            if self.debug and not finished:
                print(f"Player {self.curr+1} can't finish on an action card; drew a penalty.")
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
        # hand + discard + state(one-hot) + draw_num + open-TAKI-color(one-hot)
        #      + card shown + [direction, next player's hand size, min opponent hand size]
        if agent is None:
            agent = self.curr
        next_player = (agent + self.dir) % len(self.agents)
        other_hand_sizes = [len(h) for i, h in enumerate(self.hands) if i != agent]
        min_other = min(other_hand_sizes) if other_hand_sizes else 0
        extra = np.array([self.dir, len(self.hands[next_player]), min_other])
        return np.concatenate(
            (card_to_vector(*self.hands[agent]) if len(self.hands[agent]) > 0
             else np.zeros(CARD_VECTOR_SIZE, dtype=int),
             card_to_vector(*self.discard),
             state_to_vector(self.state),
             np.array([self.draw_num]),
             color_to_vector(self.taki_color),
             card_to_vector(self.shown_card()),
             extra))
