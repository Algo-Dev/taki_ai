import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt

from agents.dqn import AIAgent
from game import Game

# Existing checkpoints under ./models were produced by the previous (broken)
# architecture and are NOT compatible with the fixed network, so they will not
# load — training starts from fresh weights. After a run completes, point this
# at the freshly-saved checkpoint to continue training / supply trained opponents.
MODEL_PATH = None


def plot_rewards(values, wins, title=''):
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
    plt.show()


if __name__ == '__main__':
    print('Training a DQN agent via self-play against 3 opponents')
    trials = 100
    trial_len = 300
    update_target_network = 100
    num_of_players = 4

    # The learner sits at seat 0; the opponents play mostly-greedily on their own nets.
    dqn_agent = AIAgent(load_model=MODEL_PATH)
    opponents = [AIAgent(epsilon=0.1, epsilon_min=0.1, load_model=MODEL_PATH)
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

            dqn_agent.remember(state, action, reward, new_state, done)
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

    plot_rewards(rewards, wins, 'Rewards over episodes')
    dqn_agent.model.save(f'./models/checkpoint{datetime.now().timestamp()}')
