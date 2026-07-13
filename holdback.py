#!/usr/bin/env python3
"""B2 — does a policy ever hold cards back? (PLAN.md B-series)

An in-the-wild census. B1's probes were hand-built positions; its own caveat was that
they are "hand-picked, not sampled from play". This walks real games and, at every
decision, asks whether the agent chose a move that sheds FEWER cards this turn than
some legal alternative — which is exactly what `train.py`'s shaped reward (-len(hand)
per step) pays against.

The central distinction, and the one the Arm-1 ablation says is worth ~12 points:

  REFUSAL hold-back   — decline to act at all: DRAW, or CLOSE a TAKI/King continuation,
                        while a legal play exists. You keep the card AND lose the tempo.
  PREFERENCE hold-back— play a different card instead. You keep the card, spend a turn.

`max_shed` is the ground truth for "fewer cards", and it is a rules-level search rather
than a rule of thumb, because action identity lies: closing a TAKI on a PLUS or a King
KEEPS THE TURN (game.py:584-598), so CLOSE_TAKI is not always a refusal at all.
"""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')

import argparse
import collections
import copy
import random
import time

from game import (Game, Card, Action, State, Type, Color, FINISHING_TYPE_VALUES,
                  action_to_scalar)
from agents.random import RandomAgent
from agents.heuristic import HeuristicAgent, Weights, GREEDY, make_heuristic

TURN_CAP = 2000


# ---------------------------------------------------------------------------------
# Cloning
# ---------------------------------------------------------------------------------

def clone_game(game, seed=0):
    """A deep-enough copy to search on: shares the agents, rebuilds every Card.

    Rebuilding the Cards is not paranoia. `draw_card` recycles the discard and mutates
    cards IN PLACE (`card.color = Color.NONE` for a CHCOL, game.py:506-508), so a clone
    that shared Card objects could corrupt the parent's discard — including its shown
    card — mid-search. The agents list is copied (not the agents) so a forcer can be
    slotted into a seat without touching the real table.
    """
    g = copy.copy(game)
    g.agents = list(game.agents)
    g.hands = [[Card(c.type, c.color) for c in h] for h in game.hands]
    g.deck = [Card(c.type, c.color) for c in game.deck]
    g.discard = [Card(c.type, c.color) for c in game.discard]
    g.history = list(game.history)
    g.random = random.Random(seed)
    return g


class _Forcer:
    """Plays one predetermined move, then refuses to be asked again."""

    def __init__(self, move):
        self.move = move

    def play(self, game):
        assert self.move is not None, 'forcer asked twice'
        mv, self.move = self.move, None
        return mv


# ---------------------------------------------------------------------------------
# Ground truth: how many cards CAN this turn shed, starting with move `first`?
# ---------------------------------------------------------------------------------

ShedResult = collections.namedtuple('ShedResult', 'shed wins exhausted')


def max_shed(game, seat, first, budget=400):
    """Best net cards shed for the REST OF THIS TURN if `first` is played now.

    Net, so a DRAW or a finishing-rule penalty draw counts negatively. Searches the
    whole continuation (TAKI runs, King chains, PLUS follow-ups) because the turn does
    not end until the seat changes — that is what makes this correct where a hand-written
    rule would not be.
    """
    start = len(game.hands[seat])
    best = [-99]
    wins = [False]
    spent = [0]
    exhausted = [False]

    def step(g, move):
        g2 = clone_game(g)
        g2.agents[seat] = _Forcer(move)
        done, _ = g2.next_turn()
        return g2, done

    def rec(g):
        spent[0] += 1
        if spent[0] > budget:
            exhausted[0] = True
            return
        for mv in g.valid_moves():
            g2, done = step(g, mv)
            if done or g2.curr != seat:
                shed = start - len(g2.hands[seat])
                if shed > best[0]:
                    best[0] = shed
                if done and not g2.hands[seat]:
                    wins[0] = True
            else:
                rec(g2)

    g1, done1 = step(game, first)
    if done1 or g1.curr != seat:
        best[0] = start - len(g1.hands[seat])
        wins[0] = done1 and not g1.hands[seat]
    else:
        rec(g1)
    return ShedResult(best[0], wins[0], exhausted[0])


# ---------------------------------------------------------------------------------
# The classifier
# ---------------------------------------------------------------------------------

def is_refusal(game, move):
    """A move that declines to put a card down while a legal play exists.

    DRAW always is. CLOSE_TAKI is only a refusal when it actually ENDS the turn: closing
    on a PLUS or a King grants another card, so the seat keeps playing and nothing was
    refused (game.py:584-598).
    """
    action, _ = move
    if action is Action.DRAW:
        return True
    if action is Action.CLOSE_TAKI:
        if game.state is State.KING:
            return True              # declining the King's free follow-up
        top = game.shown_card()
        return top.type not in (Type.PLUS, Type.KING)
    return False


def playable_moves(game):
    """Legal PLAY moves that would not merely eat the finishing penalty.

    With one card left and no legal finisher (i.e. it is a PLUS), 'playing' sheds nothing
    — the engine hands the card straight back. Such a move is not a real alternative, so
    an agent that draws instead is not holding anything back.
    """
    hand = game.hands[game.curr]
    out = []
    for action, card in game.valid_moves():
        if action is not Action.PLAY_CARD:
            continue
        if len(hand) == 1 and card.type.value not in FINISHING_TYPE_VALUES:
            continue
        out.append((action, card))
    return out


class Census:
    def __init__(self, dfs_budget=400):
        self.dfs_budget = dfs_budget
        self.c = collections.Counter()
        self.by_state = collections.Counter()

    def observe(self, game, move):
        seat = game.curr
        moves = game.valid_moves()
        self.c['decisions'] += 1
        if len(moves) == 1:
            self.c['forced'] += 1
            return
        self.c['free'] += 1

        plays = playable_moves(game)
        if plays and is_refusal(game, move):
            # THE headline signature: it could have put a card down and did not.
            self.c['refusal'] += 1
            action, _ = move
            if action is Action.DRAW:
                self.c['refusal_draw'] += 1
            elif game.state is State.KING:
                self.c['refusal_decline_king'] += 1
            else:
                self.c['refusal_close_taki'] += 1
            self.by_state[game.state.name] += 1

        # Preference hold-back: among moves, did it pick one that sheds fewer cards
        # this turn than the best available? (Needs the search; only interesting when
        # a multi-card discharge is on the table at all.)
        if len(moves) > 1:
            sheds = {}
            exhausted = False
            for mv in moves:
                r = max_shed(game, seat, mv, self.dfs_budget)
                sheds[action_to_scalar(*mv)] = r.shed
                exhausted |= r.exhausted
            if exhausted:
                self.c['dfs_exhausted'] += 1
                return
            best = max(sheds.values())
            chosen = sheds[action_to_scalar(*move)]
            spread = best - min(sheds.values())
            if spread > 0:
                self.c['shed_choice'] += 1          # a real fewer-vs-more choice existed
                if chosen < best:
                    self.c['held_back'] += 1
                    self.c['shed_forgone'] += best - chosen
                    if not is_refusal(game, move):
                        self.c['held_back_preference'] += 1
                    else:
                        self.c['held_back_refusal'] += 1

    def report(self, label):
        c = self.c
        free = max(c['free'], 1)
        sc = max(c['shed_choice'], 1)
        print(f'\n=== {label} ===')
        print(f'  decisions {c["decisions"]}  (forced {c["forced"]}, free {c["free"]})')
        print(f'  REFUSAL hold-back (declined to play while a play was legal):')
        print(f'    total            {c["refusal"]:6d}  = {c["refusal"]/free:6.2%} of free decisions')
        print(f'      draw           {c["refusal_draw"]:6d}')
        print(f'      close TAKI     {c["refusal_close_taki"]:6d}')
        print(f'      decline King   {c["refusal_decline_king"]:6d}')
        print(f'  PREFERENCE hold-back (played, but shed fewer than it could):')
        print(f'    choices w/ a shed spread {c["shed_choice"]:6d}')
        print(f'    held back                {c["held_back"]:6d}  = {c["held_back"]/sc:6.2%} of those')
        print(f'      ...by preference       {c["held_back_preference"]:6d}')
        print(f'      ...by refusal          {c["held_back_refusal"]:6d}')
        print(f'    cards forgone            {c["shed_forgone"]:6d}')
        if c['dfs_exhausted']:
            print(f'  (dfs budget exhausted at {c["dfs_exhausted"]} nodes — excluded)')


class Recorder:
    """Duck-typed proxy. Deliberately does NOT expose reseed(): eval.play_match would
    call it and reset the wrapped heuristic's table memory at the wrong moment."""

    def __init__(self, inner, census):
        self.inner = inner
        self.census = census

    def play(self, game):
        move = self.inner.play(game)
        self.census.observe(game, move)
        return move


# ---------------------------------------------------------------------------------

def make_agent(spec):
    """The shared spec grammar (agents.heuristic.resolve_weights) — versions, ablations
    and k=v overrides alike. This file used to carry its OWN parser, which knew nothing
    about named versions: `holdback.py heuristic:h1` would have silently censused the
    R3 weights with an unknown key. One grammar, one meaning, everywhere."""
    if spec == 'random':
        return RandomAgent(seed=0)
    if spec == 'heuristic':
        return make_heuristic()
    if spec.startswith('heuristic:'):
        return make_heuristic(spec.split(':', 1)[1])
    from agents.dqn import AIAgent
    return AIAgent(epsilon=0.0, epsilon_min=0.0, load_model=spec)


def run(spec, games, seed, dfs_budget):
    """Census the agent in self-play (all four seats), which is the distribution the
    DQN was actually trained under. Every seat is recorded, so 4x the data per game."""
    agent = make_agent(spec)
    census = Census(dfs_budget)
    rec = Recorder(agent, census)
    game = Game([rec] * 4, seed=seed)
    for g in range(games):
        if hasattr(agent, 'reseed'):
            agent.reseed(f'{seed}:{g}')
        game.random.seed(seed + g)
        game.reset()
        for _ in range(TURN_CAP):
            done, _ = game.next_turn()
            if done:
                break
    return census


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('agents', nargs='+', help='agent specs (checkpoint path, random, '
                                             'heuristic, heuristic:greedy, heuristic:k=v,...)')
    p.add_argument('--games', type=int, default=300)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dfs-budget', type=int, default=400)
    args = p.parse_args()

    for spec in args.agents:
        t0 = time.time()
        census = run(spec, args.games, args.seed, args.dfs_budget)
        census.report(f'{spec}   ({args.games} games, seed {args.seed})')
        print(f'  (took {time.time() - t0:.1f}s)')


if __name__ == '__main__':
    main()
