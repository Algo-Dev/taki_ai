import argparse
import os
import random
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # headless-safe backend; switched to interactive only with --show
from matplotlib import pyplot as plt

from agents.dqn import AIAgent  # configures TF threading on import; keep before tensorflow use
import tensorflow as tf
from game import Game, action_to_scalar

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
    parser.add_argument('--color-sym', action='store_true',
                        help='augment each replayed transition with a random relabeling of '
                             'the four colors (24 TAKI color symmetries); learner only')
    parser.add_argument('--trial-len', type=int, default=300,
                        help='max learner steps per episode before the trial is cut off (default 300)')
    parser.add_argument('--target-sync-every', type=int, default=100,
                        help='hard-copy the online net into the target net every N learner steps '
                             '(default 100; legacy mode also syncs at every episode end)')
    # --- DQN-hygiene levers (default = vanilla; see RESEARCH_LOG / PLAN A1-A3) ---------------
    parser.add_argument('--target-sync-mode', choices=('legacy', 'steps', 'polyak'),
                        default='legacy',
                        help="'legacy' (default): hard-copy on step %% --target-sync-every AND at "
                             "every episode end (near-on-policy target). 'steps': hard-copy on a "
                             "GLOBAL learner-step counter %% --target-sync-every, no episode-end "
                             "copy (a genuinely lagged target). 'polyak': soft EMA update every "
                             "replay, no hard copies.")
    parser.add_argument('--polyak-tau', type=float, default=0.005,
                        help='soft-update rate for --target-sync-mode polyak (default 0.005)')
    parser.add_argument('--double-dqn', action='store_true',
                        help='Double DQN bootstrap (online net selects, target net evaluates); '
                             'learner only. Needs a lagged target (--target-sync-mode) to help.')
    parser.add_argument('--loss', choices=('mse', 'huber'), default='mse',
                        help="learner loss (default mse); 'huber' bounds per-sample gradients")
    parser.add_argument('--huber-delta', type=float, default=1.0,
                        help='delta for --loss huber (default 1.0; pair with --reward-scale so '
                             'delta sits inside the TD-error distribution)')
    parser.add_argument('--reward-scale', type=float, default=1.0,
                        help='multiply every per-step reward by this before storing it (default '
                             '1.0). NEVER combine with --model warm-start: a reward-scale change '
                             'on loaded Q-values is the win-only-collapse scale shock. Fresh runs only.')
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

    if args.reward_scale != 1.0 and args.model is not None:
        parser.error('--reward-scale with --model warm-start is the scale-shock that collapsed '
                     'the win-only run; use a fresh run (no --model) when changing reward scale.')

    # The learner sits at seat 0; the opponents play mostly-greedily on their own nets.
    # color_sym / double_dqn / loss only affect replay(), which only the learner runs, so the
    # opponents stay plain greedy nets (constructed below with defaults).
    dqn_agent = AIAgent(epsilon=args.epsilon_start, load_model=args.model,
                        color_sym=args.color_sym, double_dqn=args.double_dqn,
                        loss=args.loss, huber_delta=args.huber_delta)
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
    # The _colorsym tag keeps A/B runs distinguishable at a glance; config.txt records
    # the full arguments for later comparison.
    os.makedirs('./models', exist_ok=True)
    timestamp = datetime.now().timestamp()
    tag = '_colorsym' if args.color_sym else ''
    run_dir = f'./models/run{timestamp}{tag}'
    os.makedirs(run_dir, exist_ok=True)
    with open(f'{run_dir}/config.txt', 'w') as f:
        f.write(f'{vars(args)!r}\n')

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
    # Global learner-step counter (spans episodes). The per-episode `step` resets each trial,
    # which is why 'legacy' target syncs land near step 0 every episode (A1); 'steps' mode uses
    # this instead so the target lags by a fixed number of gradient-eligible steps.
    global_step = 0

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
        game.reset()
        episode_reward = 0
        for step in range(trial_len):
            # At the top of each step it is the learner's (seat 0) turn.
            done, _ = game.next_turn()                 # learner acts via play()
            state = dqn_agent.last_state
            action = dqn_agent.last_action
            learner_won = done and len(game.hands[0]) == 0
            # Fast-forward the opponents until it is the learner's turn again.
            while not done and game.curr != 0:
                done, _ = game.next_turn()

            new_state = game.observation(agent=0)
            if args.reward == 'anneal':
                # Curriculum: dense per-step penalty scaled by step_coef (1.0 -> floor), and on
                # a win the end reward blends from the original opponents'-card-sum toward the
                # clipped min-opponent reward (alpha 0 -> ALPHA_MAX). At p=0 this is exactly the
                # 'shaped' reward; at p=1 it is almost the 'win' reward (tiny dense floor).
                reward = step_coef * (-len(game.hands[0]))
                if learner_won:
                    opp = [len(h) for i, h in enumerate(game.hands) if i != 0]
                    reward += (1.0 - alpha) * sum(opp) + alpha * min(min(opp), 4)
            elif args.reward == 'win':
                # Win-only: no per-step / loss signal (so the agent isn't punished for
                # strategically taking a card). On a win, reward = the fewest cards any
                # opponent still holds, clipped to 4 -> a decisive win against even the
                # best-placed opponent scores highest, capped so blowouts don't dominate.
                reward = 0
                if learner_won:
                    reward = min(min(len(h) for i, h in enumerate(game.hands) if i != 0), 4)
            else:
                # Shaped (default): minus the cards the learner holds after its turn,
                # plus the sum of the opponents' cards if the learner wins.
                reward = -len(game.hands[0])
                if learner_won:
                    reward += sum(len(h) for i, h in enumerate(game.hands) if i != 0)
            # Optional global reward scaling (default 1.0 = no-op), applied uniformly to every
            # reward mode so it only rescales magnitude, not the shaping. Lets Huber's delta sit
            # inside the TD-error distribution without changing the objective.
            reward *= args.reward_scale

            # When the episode continues it is again the learner's turn (curr == 0),
            # so these are exactly the actions it may pick next — used to mask the
            # bootstrap target in replay().
            next_valid = None if done else [action_to_scalar(*m)
                                            for m in game.valid_moves(agent=0)]
            dqn_agent.remember(state, action, reward, new_state, done, next_valid)
            # replay() cadence kept as-is (every 4 steps): pairs reasonably with the larger
            # replay buffer, and the slow predict/fit makes more frequent replay costly.
            if step % 4 == 0:
                dqn_agent.replay()
                if args.target_sync_mode == 'polyak':
                    dqn_agent.polyak_update(args.polyak_tau)
            # Target sync. 'legacy': the original per-episode-step hard copy (fires near step 0
            # each episode -> near-on-policy target). 'steps': hard copy on the GLOBAL counter,
            # a genuinely lagged target. 'polyak': soft update above, no hard copy here.
            if args.target_sync_mode == 'legacy' and step % update_target_network == 0:
                dqn_agent.target_train()
            elif args.target_sync_mode == 'steps' and global_step % update_target_network == 0:
                dqn_agent.target_train()
            global_step += 1
            episode_reward += reward
            if done:
                if learner_won:
                    total_wins += 1
                break

        dqn_agent.replay()
        if args.target_sync_mode == 'polyak':
            dqn_agent.polyak_update(args.polyak_tau)
        elif args.target_sync_mode == 'legacy':
            # The unconditional episode-end hard copy — this is what made the target
            # near-on-policy (A1). 'steps' deliberately omits it; the target lags by
            # --target-sync-every global steps instead.
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
