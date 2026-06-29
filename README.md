# Taki AI
This is an experiment of teaching an AI to play [Taki](https://www.takigame.com/), a game which I own no rights to.

This project uses Keras and Tensorflow for Neural Networks.

The game does not use the 3+, 3+ breaker cards and the king card (new Taki cards).

This project taught me a lot about reinforcement learning and AI and I hope to continue making silly robots that learn!

# Algorithm
Uses DQN with experience replay.

The heuristic is minus the amount of cards a player has when he finishes his turn, and the sum of the enemy cards on win.

This proved to improve the win rate by a substantial amount, also teaching the AI to use 2+ against enemy players.

## Card Vector
Cards are a vector with dimension of the number of cards, with one at the respective index.

A deck is a vector sum of card vectors.

## Action Space
The action space takes the card scalar if it's the play card action, or draw card scalar, or close taki scalar for other instances.

## State Space
A concatenation of the following vectors / scalars:
* The hand of the player
* The discard pile
* The state
* The amount of 2+ stacked
* The card shown on the table 

# Using the project
`train.py` trains the AI.

`main.py` Let's you play against opponents. 
You can customize the agents which you play against.

Random agents are pretty easy to beat since they are likely to draw cards when they are low on cards. 

AI Agents need to be given a model to use or else they will use the random policy.

`game.py` defines all the classes needed to run the game.

`agents` a folder that includes playable agents.

`agents/dqn.py` the DQN agent.

`agents/human.py` the human agent.

`agents/random.py` an agent with a random policy.

`models` a folder that includes precomputed models. 

# License

**Non-commercial.** This project is licensed under the
[PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0)
— see [LICENSE](LICENSE). That covers the code and documentation authored by Ori Honigman **and the
trained network weights** published on this repository's releases. Personal, research, educational
and other non-commercial use is free; **commercial use is not permitted** under these terms. For a
commercial license, [open an issue](https://github.com/Algo-Dev/taki_ai/issues).

**Original code stays MIT.** Whatever survives of the starting point,
[dondish/taki-ai](https://github.com/dondish/taki-ai) (© 2020 Oded "Dondish" Shapira), remains under
the MIT License — see [LICENSE-MIT](LICENSE-MIT).
