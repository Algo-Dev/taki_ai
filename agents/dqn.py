from collections import deque

import numpy as np
import random

from tensorflow import keras
from tensorflow.keras import layers

from game import (action_to_scalar, scalar_to_action,
                  OBSERVATION_SIZE, ACTION_SIZE)

# I would like to thank https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c
# for making an easy to read tutorial on DQN with Keras, I didn't know how to implement this and it really helped.


class AIAgent:

    def __init__(self, gamma=0.99, epsilon=1.0, epsilon_min=0.1, batch_size=32,
                 epsilon_decay=0.995, learning_rate=0.01, load_model=None):
        super(AIAgent, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.memory = deque(maxlen=2000)
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        # Set by play() so a training loop can recover exactly which decision was made.
        self.last_state = None
        self.last_action = None
        self.model = self.create_model()
        self.target_model = self.create_model()
        if load_model is not None:
            try:
                loaded = keras.models.load_model(load_model)
                self.model.set_weights(loaded.get_weights())
                self.target_model.set_weights(loaded.get_weights())
            except Exception as e:
                print(f"Warning: could not load model from '{load_model}': {e}. "
                      f"Using freshly initialised weights.")

    def create_model(self):
        model = keras.Sequential()
        model.add(layers.Dense(124, input_dim=OBSERVATION_SIZE, activation="relu"))
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(ACTION_SIZE))
        model.compile(loss="mean_squared_error",
                      optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        samples = random.sample(self.memory, self.batch_size)
        states = np.array([s[0] for s in samples])          # (batch, OBSERVATION_SIZE)
        next_states = np.array([s[3] for s in samples])     # (batch, OBSERVATION_SIZE)
        targets = self.model.predict(states)                # (batch, ACTION_SIZE)
        next_q = self.target_model.predict(next_states)     # (batch, ACTION_SIZE)
        for i, (_, action, reward, _, done) in enumerate(samples):
            if done:
                targets[i][action] = reward
            else:
                targets[i][action] = reward + self.gamma * np.max(next_q[i])
        self.model.fit(states, targets, epochs=1, verbose=0, batch_size=self.batch_size)

    def target_train(self):
        self.target_model.set_weights(self.model.get_weights())

    def decay_epsilon(self):
        """Decay exploration once per environment step (called by the training loop)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def act(self, state, actions):
        if np.random.random() < self.epsilon:
            return random.choice(actions)
        q_values = self.model.predict(state[np.newaxis, :])[0]   # (ACTION_SIZE,)
        return actions[int(np.argmax(q_values[actions]))]

    def play(self, game):
        state = game.observation()
        actions = list(map(lambda x: action_to_scalar(*x), game.valid_moves()))
        action = self.act(state, actions)
        self.last_state = state
        self.last_action = action
        return scalar_to_action(action)
