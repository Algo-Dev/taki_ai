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
  number, so no action card's effect is silently dropped at the start). This is a **house-rule
  simplification**, not official Taki — the official rules just flip the top card as-is.

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
     CLOSE_TAKI action; no plain DRAW is offered during the King continuation. The follow-up
     card is processed in a **normal context, so its own effect applies and can cascade**: a
     follow-up TAKI opens a full run, a follow-up PLUS grants yet another card, a follow-up +2
     lands on the next player, and a follow-up King chains (see below). In practice this is
     close to the official "another free turn"; the house reading is only that it is optional.
- **Inside an open TAKI** the King is playable but **inert** — like every other action card
  mid-run, only the last card's effect applies. A King played mid-sequence does **not** change
  the active color and does **not** start a follow-up; the TAKI simply continues. If the TAKI
  is **closed on** a King (the King is the top card at close), the player is then granted the
  optional follow-up turn (mirrors how +2/STOP/etc. defer their effect to close).
- **Kings chain** — a King may itself be the follow-up to a King, each granting one more card.
- **You can win on a King** — like every card except PLUS (see Finishing). Note the asymmetry with
  the opening: the round may not *open* on a King; opening is numbers-only.

## Finishing (which card may be your last)

**You may end the round on any card except PLUS.** PLUS is the sole exception because it obliges
you to put one more card, and an empty hand cannot. Every other card's effect lands on someone
else and is coherent as a final play: a last STOP skips the next player, a last +2 makes them
draw, a last CHDIR reverses, a last Change Color / King / TAKI / Super TAKI simply wins (a TAKI
run you never get to use is still a legal last card).

Enforcement is a single check in `next_turn` ([game.py:694](game.py#L694)) against
`FINISHING_TYPE_VALUES`. Playing a PLUS as your last card is **legal** — it just doesn't win: the
card's own effect still applies, you draw one penalty card, and play continues (you keep the turn,
since PLUS grants you another). You only truly finish that way if the deck is exhausted and no
penalty card can be drawn.

Because PLUS is the only restriction, **sequencing a TAKI run is unconstrained unless the run
contains a PLUS** — dump a color group in any order and the last card wins. (History: the engine
formerly allowed finishing only on a number or the King. That was a **bug, not a house rule**; it
was fixed on 2026-07-12 and it invalidated the sequencing half of the B1 probe. See
`probes/b1_colored_taki_hoard.md`.)

## House-rule interpretations (deliberate; official rules ambiguous or simplified)

- A colorless Change Color is playable inside an open TAKI (it passes the color filter as a
  wild). Defensible reading; official rules are ambiguous.
- Playing a Change Color inside an open TAKI does **not** change `taki_color` for the rest of
  the TAKI — the chosen color takes effect only after the TAKI closes (the shown card then
  carries the new color).
- 2-player CHDIR is a pure no-op (`(curr+1) % 2 == (curr-1) % 2`) — and this **matches official
  Taki**: the "gives an extra turn with two players" reading is a common misconception (called
  out as wrong in the Wikipedia article). Irrelevant to the 4-player experiments either way.
- **Announcement rules are omitted** (human-attention mechanics with no analog for an agent):
  - **"Last card!"** — officially a player down to one card must announce it or draw 4 as a
    penalty. Not implemented; there is no announcement concept in the engine.
  - **Open-TAKI inheritance** — officially, failing to declare "Closed TAKI!" leaves the run
    open and the *next player* may continue dumping that color into it. Not implemented:
    CLOSE_TAKI is a mandatory explicit action by the same player before the turn passes, so an
    inherited open run can never occur (and deliberately leaving a TAKI open is not possible).
- The King's follow-up card is optional (not mandatory), and the King is inert inside a TAKI
  (see the King section) — both are deliberate readings of the "put a card after it" rule.
