# Taki rules — as implemented in `game.py`

This is the reference for how the rules engine actually behaves, including the deliberate
house-rule interpretations where official Taki is ambiguous or where a simplification was
chosen. `game.py`'s `valid_moves()` is the legality source of truth; this doc explains the
intent behind it. **Check here (not PLAN.md) before "fixing" any rules behaviour.**

## Deck

Standard Taki deck as built in `Game._setup_round`:
- Each colored type (TAKI, numbers 1–9, STOP, Change Direction, +2, PLUS) × 4 colors × 2 copies.
- 2 colorless Super TAKI, 4 colorless Change Color, **2 colorless King**.
- The round must **open on a plain number card** (the opening draw is repeated until it is a
  number, so no action card's effect is silently dropped at the start).

## Special cards

- **STOP** — skips the next player.
- **Change Direction (CHDIR)** — reverses turn order.
- **+2 (PLUSTWO)** — the next player must either stack another +2 (raising the pending draw)
  or draw `2 × (number of stacked +2s)`. A King also cancels it (see below).
- **PLUS** — the player takes one more turn (one extra card, color/type-matched as normal).
- **TAKI (colored)** — opens a run: the same player keeps playing cards of the TAKI's color
  (plus colorless wilds) until they CLOSE the TAKI. Inside an open TAKI only the **last**
  card's effect applies — every action card played mid-run is inert and its effect is realized
  at close, based on the top card.
- **Super TAKI (colorless)** — a TAKI that adopts the color of the card beneath it.
- **Change Color (CHCOL)** — colorless wild; the player picks the new active color.
- **King** — see the dedicated section below.

## The King

A colorless wild (2 copies). It can be played on any card. Its behaviour depends on context:

- **Outside a TAKI** it has two powers:
  1. **Cancels a pending +2** — playing a King in response to a +2 zeroes the accumulated
     draw (`draw_num → 0`); no cards are drawn.
  2. **Grants one optional follow-up card** — the same player may then put **one** more card of
     **any** color/type, or decline. Declining (and the "done" terminator generally) reuses the
     CLOSE_TAKI action; no plain DRAW is offered during the King continuation.
- **Inside an open TAKI** the King is playable but **inert** — like every other action card
  mid-run, only the last card's effect applies. A King played mid-sequence does **not** change
  the active color and does **not** start a follow-up; the TAKI simply continues. If the TAKI
  is **closed on** a King (the King is the top card at close), the player is then granted the
  optional follow-up turn (mirrors how +2/STOP/etc. defer their effect to close).
- **Kings chain** — a King may itself be the follow-up to a King, each granting one more card.
- **You can win on a King** — unlike other action cards, the King is a legal finishing card. (The
  round still may not *open* on a King; that is numbers-only.)

## House-rule interpretations (deliberate; official rules ambiguous or simplified)

- A colorless Change Color is playable inside an open TAKI (it passes the color filter as a
  wild). Defensible reading; official rules are ambiguous.
- Playing a Change Color inside an open TAKI does **not** change `taki_color` for the rest of
  the TAKI — the chosen color takes effect only after the TAKI closes (the shown card then
  carries the new color).
- 2-player CHDIR is a pure no-op (`(curr+1) % 2 == (curr-1) % 2`), whereas official Taki treats
  change-direction as a stop/extra-turn with two players. Irrelevant to the 4-player
  experiments; fix only if 2-player play ever matters.
- The King's follow-up card is optional (not mandatory), and the King is inert inside a TAKI
  (see the King section) — both are deliberate readings of the "put a card after it" rule.
