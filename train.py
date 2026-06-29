import argparse
import os
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # headless-safe backend; switched to interactive only with --show
from matplotlib import pyplot as plt

from agents.dqn import AIAgent
from game import Game, action_to_scalar

# How often (in trials) to copy the learner's weights into the opponents so that
# self-play actually faces a progressively stronger version of itself. The cadence
# is a deliberately simple knob; finer tuning is part of the deferred RL work.
OPPONENT_SYNC_EVERY = 5


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
    ax[1].legend()

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
    parser.add_argument('--show', action='store_true',
                        help='display the training plot interactively (otherwise only saved to PNG)')
    args = parser.parse_args()

    print('Training a DQN agent via self-play against 3 opponents')
    trials = args.trials
    trial_len = 300
    update_target_network = 100
    num_of_players = 4

    # The learner sits at seat 0; the opponents play mostly-greedily on their own nets.
    dqn_agent = AIAgent(load_model=args.model)
    opponents = [AIAgent(epsilon=0.1, epsilon_min=0.1, load_model=args.model)
                 for _ in range(num_of_players - 1)]
    game = Game([dqn_agent, *opponents])

    rewards = []
    wins = []
    total_wins = 0

    for trial in range(trials):
        print(f"Trial {trial + 1}/{trials}")
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
            # Heuristic reward: minus the cards the learner holds after its turn,
            # plus the sum of the opponents' cards if the learner wins.
            reward = -len(game.hands[0])
            if learner_won:
                reward += sum(len(h) for i, h in enumerate(game.hands) if i != 0)

            # When the episode continues it is again the learner's turn (curr == 0),
            # so these are exactly the actions it may pick next — used to mask the
            # bootstrap target in replay().
            next_valid = None if done else [action_to_scalar(*m)
                                            for m in game.valid_moves(agent=0)]
            dqn_agent.remember(state, action, reward, new_state, done, next_valid)
            dqn_agent.decay_epsilon()
            if step % 4 == 0:
                dqn_agent.replay()
            if step % update_target_network == 0:
                dqn_agent.target_train()
            episode_reward += reward
            if done:
                if learner_won:
                    total_wins += 1
                break

        dqn_agent.replay()
        dqn_agent.target_train()
        rewards.append(episode_reward)
        wins.append(total_wins)

        # Real self-play: periodically promote the learner's weights into the
        # opponents so it keeps facing a stronger version of itself.
        if (trial + 1) % OPPONENT_SYNC_EVERY == 0:
            learner_weights = dqn_agent.model.get_weights()
            for opp in opponents:
                opp.model.set_weights(learner_weights)
                opp.target_model.set_weights(learner_weights)

    os.makedirs('./models', exist_ok=True)
    timestamp = datetime.now().timestamp()
    plot_rewards(rewards, wins, 'Rewards over episodes',
                 save_path=f'./models/training{timestamp}.png', show=args.show)
    dqn_agent.model.save(f'./models/checkpoint{timestamp}')
