import argparse
import random
import time

from game import Game
from agents.random import RandomAgent
from agents.human import HumanAgent
from agents.dqn import AIAgent

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Play / demo Taki agents.')
    # Pass a checkpoint from a current-architecture training run to use trained
    # weights. The old ./models/checkpoint* dirs are incompatible and would fall
    # back to random weights (with a warning). Leave unset for random play.
    parser.add_argument('--model', default=None,
                        help='checkpoint to load the DQN agents from')
    args = parser.parse_args()

    random.seed(42)  # Set seed for reproducibility

    # Normal game, 3 DQN agents (add HumanAgent() to play along).
    agents = [AIAgent(load_model=args.model),
              AIAgent(load_model=args.model),
              AIAgent(load_model=args.model)]  # , HumanAgent()]
    random.shuffle(agents)
    game = Game(agents, True)
    print(game.observation().shape)

    games_won_per_player = [0] * len(agents)
    num_iterations = 4
    total_time = 0

    for i in range(num_iterations):
        start_time = time.time()  # Start time for iteration

        while not game.done():
            game.next_turn()
        winner_idxs = [idx for idx, hand in enumerate(game.hands) if not hand]
        if winner_idxs:
            games_won_per_player[winner_idxs[0]] += 1
        game.reset()

        iteration_time = time.time() - start_time  # Calculate iteration time
        total_time += iteration_time
        average_time = total_time / (i + 1)
        remaining_iterations = num_iterations - (i + 1)
        estimated_time_remaining = average_time * remaining_iterations

        print(f"Iteration {i+1}/{num_iterations} completed in {iteration_time:.2f} seconds.")
        print(f"Total run time: {total_time:.2f} seconds. Average iteration time: {average_time}")
        print(f"Estimated time remaining: {estimated_time_remaining:.2f} seconds.")

    print(f"Games won per player: {games_won_per_player}")
