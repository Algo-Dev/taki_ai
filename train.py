import argparse
import os
import random
import sys
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # headless-safe backend; switched to interactive only with --show
from matplotlib import pyplot as plt

from agents.dqn import AIAgent  # configures TF threading on import; keep before tensorflow use
import tensorflow as tf
from game import Game, action_to_scalar

class _Tee:
    """Mirror stdout into run_dir/train.log so a run's progress survives the process.

    Runs take hours and have twice been lost to a WSL2 VM wedge; a shell redirect into
    /tmp did not survive, because systemd-tmpfiles empties /tmp at boot (`D /tmp ...`).
    Writing beside the snapshots keeps the log for the post-mortem. Flushed per write:
    a killed run must leave its last trial number on disk, not in a buffer.
    """

    def __init__(self, stream, path):
        self.stream = stream
        self.file = open(path, 'a', buffering=1)

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()
        return len(data)

    def flush(self):
        self.stream.flush()
        self.file.flush()


# How often (in trials) to copy the learner's weights into the opponents so that
# self-play actually faces a progressively stronger version of itself. The cadence
# is a deliberately simple knob; finer tuning is part of the deferred RL work.
OPPONENT_SYNC_EVERY = 5

# Epsilon decays once per episode (not per step), reaching epsilon_min after this
# fraction of the trials so exploration lasts across most of training rather than
# bottoming out in the first couple of episodes.
EPSILON_DECAY_FRACTION = 0.8

# How often (in trials) to save an intermediate model snapshot, so a later eval can
# measure improvement of the latest model over earlier ones (see eval.py mode B).
SNAPSHOT_EVERY = 25

# --reward anneal: gradually drift the shaped reward toward (almost) win-only over the first
# REWARD_ANNEAL_FRACTION of trials, then hold. An instantaneous switch collapsed the policy
# (see RESEARCH_LOG.md); annealing lets the value function track a slowly-moving target. At
# progress p in [0,1]: the dense per-step penalty is scaled 1.0 -> STEP_COEF_FLOOR, and the
# end-of-game reward blends from the original opponents'-card-sum toward the clipped
# min-opponent reward with weight alpha 0 -> ALPHA_MAX.
REWARD_ANNEAL_FRACTION = 0.8
ALPHA_MAX = 0.99
STEP_COEF_FLOOR = 0.01


def plot_rewards(values, wins, title='', save_path=None, show=False):
    f, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
    f.suptitle(title)
    ax[0].plot(values, label='reward per episode')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Reward')
    x = range(len(values))
    ax[0].legend()
    # Calculate the trend
    try:
        z = np.polyfit(x, values, 1)
        p = np.poly1d(z)
        ax[0].plot(x, p(x), "--", label='trend')
    except (np.linalg.LinAlgError, TypeError, ValueError):
        pass

    # Plot the histogram of results
    ax[1].hist(values[-50:])
    ax[1].set_xlabel('Rewards per Last 50 Episodes')
    ax[1].set_ylabel('Frequency')

    ax[2].plot(wins, label='accumulated wins')
    ax[2].set_xlabel('Episode')
    ax[2].set_ylabel('Accumulated wins')
    ax[2].legend()
    if save_path is not None:
        f.savefig(save_path)
        print(f'Saved training plot to {save_path}')
    if show:
        plt.show()
    plt.close(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a DQN Taki agent via self-play.')
    # The old ./models/checkpoint* dirs were produced by a previous architecture and
    # will NOT load against the current network — leave --model unset to start fresh,
    # or pass a checkpoint from a current-architecture run to continue training.
    parser.add_argument('--model', default=None,
                        help='checkpoint to warm-start the learner and opponents from')
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--epsilon-start', type=float, default=1.0,
                        help='initial exploration rate (default 1.0). Lower it (down to '
                             'epsilon_min) to continue a warm-started model near-greedily '
                             'instead of re-exploring from scratch.')
    parser.add_argument('--snapshot-every', type=int, default=SNAPSHOT_EVERY,
                        help=f'save an intermediate snapshot every N trials (default {SNAPSHOT_EVERY})')
    parser.add_argument('--reward', choices=('shaped', 'win', 'anneal'), default='shaped',
                        help="reward shaping: 'shaped' = per-step -len(hand) + win bonus "
                             "(default); 'win' = win-only, no per-step penalty; 'anneal' = "
                             "curriculum drifting shaped -> (almost) win-only over training.")
    parser.add_argument('--reward-anneal-fraction', type=float, default=REWARD_ANNEAL_FRACTION,
                        help='for --reward anneal: fraction of trials over which the reward '
                             f'transitions, then holds (default {REWARD_ANNEAL_FRACTION})')
    parser.add_argument('--no-color-sym', dest='color_sym', action='store_false',
                        help='disable color-symmetry augmentation (on by default): each '
                             'replayed transition is trained under a random relabeling of the '
                             'four colors (24 TAKI color symmetries); learner only. Off = '
                             'vanilla replay, which diverges on long runs (see RESEARCH_LOG.md)')
    parser.set_defaults(color_sym=True)
    parser.add_argument('--trial-len', type=int, default=300,
                        help='max learner steps per episode before the trial is cut off (default 300)')
    parser.add_argument('--target-sync-every', type=int, default=100,
                        help='hard-copy the online net into the target net every N learner steps '
                             '(default 100; note the loop also syncs at every episode end)')
    parser.add_argument('--seed', type=int, default=None,
                        help='seed Python/NumPy RNGs and the game deck for a reproducible run '
                             '(default: unseeded)')
    parser.add_argument('--show', action='store_true',
                        help='display the training plot interactively (otherwise only saved to PNG)')
    args = parser.parse_args()

    # Seed everything the run touches so replicate runs are bit-reproducible: Python's random
    # (agent epsilon draws, replay sampling), NumPy (color-sym permutation draws, act()), TF
    # (network weight initialisation — set before the AIAgents build their models below), and
    # the game's own deck RNG (passed to Game below). Left unseeded when --seed is omitted.
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

    print(f'Training a DQN agent via self-play against 3 opponents '
          f'(reward={args.reward}, color_sym={args.color_sym})')
    trials = args.trials
    trial_len = args.trial_len
    update_target_network = args.target_sync_every
    num_of_players = 4

    # The learner sits at list index 0; the opener is randomised per trial (A4), so the learner
    # no longer goes first 100% of the time. The opponents play mostly-greedily on their own nets.
    # color_sym only affects replay(), which only the learner runs.
    dqn_agent = AIAgent(epsilon=args.epsilon_start, load_model=args.model,
                        color_sym=args.color_sym)
    if not dqn_agent.epsilon_min <= args.epsilon_start <= 1.0:
        parser.error(f'--epsilon-start must be in [{dqn_agent.epsilon_min}, 1.0]')
    # Per-episode decay sized to the run: epsilon falls from epsilon_start to epsilon_min over
    # the first EPSILON_DECAY_FRACTION of the trials (scales with --trials). If epsilon_start
    # already equals epsilon_min the ratio is 1, so epsilon stays flat (a near-greedy continue).
    dqn_agent.epsilon_decay = (dqn_agent.epsilon_min / args.epsilon_start) ** (
        1.0 / (EPSILON_DECAY_FRACTION * trials))
    opponents = [AIAgent(epsilon=0.1, epsilon_min=0.1, load_model=args.model)
                 for _ in range(num_of_players - 1)]
    game = Game([dqn_agent, *opponents], seed=args.seed)

    # One run directory shared by all snapshots and the final checkpoint/plot, so they
    # carry the same timestamp and eval.py can discover the whole progression at once.
    # Color-sym is the default now, so tag only the ablation (--no-color-sym) runs to keep
    # them distinguishable at a glance; config.txt records the full arguments regardless.
    os.makedirs('./models', exist_ok=True)
    timestamp = datetime.now().timestamp()
    tag = '' if args.color_sym else '_nocolorsym'
    run_dir = f'./models/run{timestamp}{tag}'
    os.makedirs(run_dir, exist_ok=True)
    with open(f'{run_dir}/config.txt', 'w') as f:
        f.write(f'{vars(args)!r}\n')
    sys.stdout = _Tee(sys.stdout, f'{run_dir}/train.log')
    print(f'run_dir: {run_dir}  (progress mirrored to {run_dir}/train.log)')

    def save_snapshot(trial_idx):
        """Save the learner's current weights as snap<NNNN> (zero-padded trial index)."""
        dqn_agent.model.save(f'{run_dir}/snap{trial_idx:04d}')

    # snap0000 is the starting network (untrained on a fresh run, or the warm-start
    # model when --model is given) — the baseline eval.py mode B measures every
    # later snapshot against.
    save_snapshot(0)

    rewards = []
    wins = []
    total_wins = 0

    # Dedicated opener RNG (A4). Separate from the deck RNG (game.random) and the agents'
    # epsilon/replay streams, so existing seeded decks stay bit-reproducible.
    seat_rng = random.Random(args.seed)

    # Seat -> agent map for the all-seats collection (A8). Every seat plays the same DQN
    # policy (opponents synced to the learner every OPPONENT_SYNC_EVERY trials), and the
    # observation is egocentric, so every seat's transitions are valid learner training
    # data. They all feed the learner's single replay buffer.
    agents = [dqn_agent, *opponents]

    for trial in range(trials):
        print(f"Trial {trial + 1}/{trials}")
        # Reward-anneal schedule for this trial (constant within the trial). p ramps 0->1 over
        # the first reward_anneal_fraction of trials, then holds; step_coef 1.0->floor scales
        # the dense per-step penalty, alpha 0->ALPHA_MAX blends the end-of-game reward.
        step_coef, alpha = 1.0, 0.0
        if args.reward == 'anneal':
            p = min(1.0, trial / max(1.0, args.reward_anneal_fraction * trials))
            step_coef = 1.0 - (1.0 - STEP_COEF_FLOOR) * p
            alpha = ALPHA_MAX * p
            if trial % 500 == 0:
                print(f"  [anneal] p={p:.3f} step_coef={step_coef:.3f} alpha={alpha:.3f}")
        def seat_reward(seat, won):
            # Reward for `seat`'s transition, generalised from the seat-0-only rewards to any
            # seat (the shaped/win/anneal shaping is seat-symmetric — everyone wants fewer
            # cards / to win). `won` is True only when the game just ended with `seat`'s hand
            # empty. step_coef/alpha are this trial's anneal schedule (constants here).
            if args.reward == 'anneal':
                # Curriculum: dense per-step penalty scaled by step_coef (1.0 -> floor), and on
                # a win the end reward blends from the original opponents'-card-sum toward the
                # clipped min-opponent reward (alpha 0 -> ALPHA_MAX). At p=0 this is exactly the
                # 'shaped' reward; at p=1 it is almost the 'win' reward (tiny dense floor).
                r = step_coef * (-len(game.hands[seat]))
                if won:
                    opp = [len(h) for i, h in enumerate(game.hands) if i != seat]
                    r += (1.0 - alpha) * sum(opp) + alpha * min(min(opp), 4)
                return r
            if args.reward == 'win':
                # Win-only: no per-step / loss signal (so the agent isn't punished for
                # strategically taking a card). On a win, reward = the fewest cards any
                # opponent still holds, clipped to 4 -> a decisive win against even the
                # best-placed opponent scores highest, capped so blowouts don't dominate.
                if won:
                    return min(min(len(h) for i, h in enumerate(game.hands) if i != seat), 4)
                return 0
            # Shaped (default): minus the cards the seat holds after its turn, plus the sum of
            # the opponents' cards if it wins.
            r = -len(game.hands[seat])
            if won:
                r += sum(len(h) for i, h in enumerate(game.hands) if i != seat)
            return r

        start = seat_rng.randrange(num_of_players)   # random opener; learner stays index 0
        game.reset(start_seat=start)
        episode_reward = 0
        # All-seats collection (A8): one open transition per seat, closed when that seat is
        # about to act again (below) or when the game ends (terminal loop after the round).
        # This reproduces the old seat-0 transitions exactly and adds the other three seats.
        pending = [None] * num_of_players            # pending[i] = (state, action, cur_valid)
        learner_steps = 0                            # counts seat-0 decisions (replay cadence)
        done = False
        # Turn-by-turn loop over whichever seat is to act. A seat may act several times in a
        # row (open TAKI / PLUS / KING keep curr), each captured as its own transition — same
        # as the old loop treated the learner's multi-card turns.
        while not done and learner_steps < trial_len:
            seat = game.curr
            if pending[seat] is not None:
                # This seat is about to act again: close its previous transition. new_state /
                # reward / next_valid are measured at this same moment the old loop used.
                s, a, cv = pending[seat]
                nxt = [action_to_scalar(*m) for m in game.valid_moves(agent=seat)]
                r = seat_reward(seat, won=False)
                dqn_agent.remember(s, a, r, game.observation(agent=seat), False, nxt,
                                   cur_valid=cv)
                if seat == 0:
                    episode_reward += r
                pending[seat] = None
            done, _ = game.next_turn()               # seat acts via play() (one card)
            ag = agents[seat]
            pending[seat] = (ag.last_state, ag.last_action, ag.last_valid)
            if seat == 0:
                # replay()/target_train() cadence kept as-is (keyed to learner decisions): the
                # slow predict/fit makes more frequent replay costly, and with ~4x the data per
                # trial the replay ratio already drops from ~20 to ~5 (near the Atari ~8).
                if learner_steps % 4 == 0:
                    dqn_agent.replay()
                if learner_steps % update_target_network == 0:
                    dqn_agent.target_train()
                learner_steps += 1

        # Round over (a win) or cut off by trial_len — close every still-open transition.
        for i in range(num_of_players):
            if pending[i] is None:
                continue
            s, a, cv = pending[i]
            if done:
                won = len(game.hands[i]) == 0
                r = seat_reward(i, won)
                # Terminal: new_state is unused (target == reward), next_valid None.
                dqn_agent.remember(s, a, r, game.observation(agent=i), True, None,
                                   cur_valid=cv)
                if i == 0:
                    episode_reward += r
                    if won:
                        total_wins += 1
            else:
                # trial_len cutoff (very rare at 300 learner steps): close non-terminally so no
                # acted transition is dropped. new_state at the current position is a slightly
                # approximate next-state, acceptable on this rare path.
                nxt = [action_to_scalar(*m) for m in game.valid_moves(agent=i)]
                r = seat_reward(i, won=False)
                dqn_agent.remember(s, a, r, game.observation(agent=i), False, nxt,
                                   cur_valid=cv)
                if i == 0:
                    episode_reward += r

        dqn_agent.replay()
        dqn_agent.target_train()
        dqn_agent.decay_epsilon()  # decay exploration once per episode
        rewards.append(episode_reward)
        wins.append(total_wins)

        # Real self-play: periodically promote the learner's weights into the
        # opponents so it keeps facing a stronger version of itself.
        if (trial + 1) % OPPONENT_SYNC_EVERY == 0:
            learner_weights = dqn_agent.model.get_weights()
            for opp in opponents:
                opp.model.set_weights(learner_weights)
                opp.target_model.set_weights(learner_weights)

        # Periodic snapshot for the progression eval (snap0000 was the untrained net).
        if (trial + 1) % args.snapshot_every == 0:
            save_snapshot(trial + 1)

    plot_rewards(rewards, wins, 'Rewards over episodes',
                 save_path=f'./models/training{timestamp}{tag}.png', show=args.show)
    dqn_agent.model.save(f'./models/checkpoint{timestamp}{tag}')
