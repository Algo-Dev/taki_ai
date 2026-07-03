from collections import deque

import numpy as np
import random

import tensorflow as tf
# This is a tiny network driven by single-sample / batch-64 ops, so TF's default
# thread pools (one op spread over every core) just thrash — system time and context
# switches dwarf the actual compute. A small pool is markedly faster here. Must be set
# before any TF op runs; harmless if the context is already initialised.
try:
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except RuntimeError:
    pass
from tensorflow import keras
from tensorflow.keras import layers

from game import (action_to_scalar, scalar_to_action,
                  OBSERVATION_SIZE, ACTION_SIZE)

# I would like to thank https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c
# for making an easy to read tutorial on DQN with Keras, I didn't know how to implement this and it really helped.


class AIAgent:

    def __init__(self, gamma=0.99, epsilon=1.0, epsilon_min=0.1, batch_size=64,
                 epsilon_decay=0.995, learning_rate=0.001, load_model=None):
        super(AIAgent, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.memory = deque(maxlen=20000)
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        # Set by play() so a training loop can recover exactly which decision was made.
        self.last_state = None
        self.last_action = None
        self.model = self.create_model()
        self.target_model = self.create_model()
        if load_model is not None:
            # A failed load must stop the process: silently falling back to random weights
            # turns an intended warm-start into a cold-start without anyone noticing.
            try:
                loaded = keras.models.load_model(load_model)
            except Exception as e:
                raise RuntimeError(f"could not load model from '{load_model}': {e}") from e
            self.model.set_weights(loaded.get_weights())
            self.target_model.set_weights(loaded.get_weights())

    def create_model(self):
        model = keras.Sequential()
        model.add(layers.Dense(124, input_dim=OBSERVATION_SIZE, activation="relu"))
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(ACTION_SIZE))
        model.compile(loss="mean_squared_error",
                      optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, new_state, done, next_valid=None):
        # next_valid: the action scalars that are legal in new_state, used to mask the
        # bootstrap target so it never relies on the Q-value of an illegal action.
        self.memory.append([state, action, reward, new_state, done, next_valid])

    @tf.function
    def _train_step(self, states, targets):
        # Compiled single gradient step (replaces model.fit, which retraces per call in
        # eager and carries heavy Python overhead). MSE over the full action vector — the
        # non-taken actions have target == current prediction, so they contribute no
        # gradient, exactly as the previous fit-based update did.
        with tf.GradientTape() as tape:
            preds = self.model(states, training=True)
            loss = tf.reduce_mean(tf.square(targets - preds))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        samples = random.sample(self.memory, self.batch_size)
        states = np.array([s[0] for s in samples], dtype=np.float32)       # (batch, OBS)
        next_states = np.array([s[3] for s in samples], dtype=np.float32)  # (batch, OBS)
        # Direct model() calls instead of model.predict() — far less per-call overhead
        # for batches this small.
        targets = self.model(states, training=False).numpy()       # (batch, ACTION_SIZE)
        next_q = self.target_model(next_states, training=False).numpy()
        for i, (_, action, reward, _, done, next_valid) in enumerate(samples):
            if done:
                targets[i][action] = reward
            else:
                # Bootstrap only from actions that are legal in the next state; the
                # network is never trained on illegal actions, so their Q-values are junk.
                best_next = np.max(next_q[i][next_valid]) if next_valid else np.max(next_q[i])
                targets[i][action] = reward + self.gamma * best_next
        self._train_step(tf.convert_to_tensor(states),
                          tf.convert_to_tensor(targets, dtype=tf.float32))

    def target_train(self):
        self.target_model.set_weights(self.model.get_weights())

    def decay_epsilon(self):
        """Decay exploration once per episode (called by the training loop)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def act(self, state, actions):
        if np.random.random() < self.epsilon:
            return random.choice(actions)
        # Direct model() call (not model.predict) — this runs once per turn for every
        # agent, so its per-call overhead dominates the self-play loop.
        x = state[np.newaxis, :].astype(np.float32)
        q_values = self.model(x, training=False).numpy()[0]      # (ACTION_SIZE,)
        return actions[int(np.argmax(q_values[actions]))]

    def play(self, game):
        state = game.observation()
        actions = list(map(lambda x: action_to_scalar(*x), game.valid_moves()))
        action = self.act(state, actions)
        self.last_state = state
        self.last_action = action
        return scalar_to_action(action)
