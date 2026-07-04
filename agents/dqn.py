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
                  OBSERVATION_SIZE, ACTION_SIZE, OBS_PERMS, ACT_PERMS)

# I would like to thank https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c
# for making an easy to read tutorial on DQN with Keras, I didn't know how to implement this and it really helped.


class AIAgent:

    def __init__(self, gamma=0.99, epsilon=1.0, epsilon_min=0.1, batch_size=64,
                 epsilon_decay=0.995, learning_rate=0.001, load_model=None,
                 color_sym=False, double_dqn=False, loss='mse', huber_delta=1.0):
        super(AIAgent, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        # Replay-time color-symmetry augmentation: TAKI's colors are interchangeable,
        # so each sampled transition is trained under a random relabeling of the four
        # colors (see the COLOR_PERMS tables in game.py).
        self.color_sym = color_sym
        # DQN-hygiene levers (all default to vanilla behavior so an off run is bit-identical
        # to the pre-flag code):
        #   double_dqn  - select the bootstrap action with the ONLINE net, evaluate it with
        #                 the TARGET net (decouples selection from evaluation -> less
        #                 overestimation). Needs a lagged target to help (see train.py's
        #                 --target-sync-mode); with an on-policy target it degenerates to max.
        #   loss/delta  - 'huber' bounds the per-sample gradient at +/-delta, capping the
        #                 influence of rare large TD errors (wins, stale bootstraps). Pair
        #                 with a reward scale that puts delta inside the error distribution.
        self.double_dqn = double_dqn
        self.loss = loss
        self.huber_delta = huber_delta
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
        # eager and carries heavy Python overhead). Loss is over the full action vector —
        # the non-taken actions have target == current prediction, so they contribute no
        # gradient, exactly as the previous fit-based update did. self.loss is a Python
        # attribute fixed at construction, so this branch is resolved once at trace time
        # (no per-call retrace); the compiled graph is MSE or Huber for the agent's life.
        with tf.GradientTape() as tape:
            preds = self.model(states, training=True)
            err = targets - preds
            if self.loss == 'huber':
                # Quadratic within +/-delta, linear beyond -> bounded gradient magnitude.
                a = tf.minimum(tf.abs(err), self.huber_delta)
                loss = tf.reduce_mean(0.5 * a * a + self.huber_delta * (tf.abs(err) - a))
            else:
                loss = tf.reduce_mean(tf.square(err))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        samples = random.sample(self.memory, self.batch_size)
        states = np.array([s[0] for s in samples], dtype=np.float32)       # (batch, OBS)
        next_states = np.array([s[3] for s in samples], dtype=np.float32)  # (batch, OBS)
        if self.color_sym:
            # One uniformly-random color relabeling per transition (identity included);
            # state, new_state, action and next_valid all get the SAME permutation, so
            # each row stays a genuine environment transition (reward/done are
            # color-invariant). Must happen BEFORE the model() calls below: the target
            # rows are the predictions on the augmented states, so the 63 untouched
            # entries keep target == prediction (zero gradient). The buffer keeps the
            # originals — states/next_states are fresh copies and take_along_axis
            # allocates new arrays.
            ks = np.random.randint(len(OBS_PERMS), size=self.batch_size)
            states = np.take_along_axis(states, OBS_PERMS[ks], axis=1)
            next_states = np.take_along_axis(next_states, OBS_PERMS[ks], axis=1)
        # Direct model() calls instead of model.predict() — far less per-call overhead
        # for batches this small.
        targets = self.model(states, training=False).numpy()       # (batch, ACTION_SIZE)
        next_q = self.target_model(next_states, training=False).numpy()
        # Double DQN: the ONLINE net selects the bootstrap action, the TARGET net evaluates
        # it. One extra batched forward pass, only when the flag is on. Uses the same
        # (possibly color-permuted) next_states as next_q, so the two agree row-for-row.
        online_next = (self.model(next_states, training=False).numpy()
                       if self.double_dqn else None)
        for i, (_, action, reward, _, done, next_valid) in enumerate(samples):
            if self.color_sym:
                act_f = ACT_PERMS[ks[i]]
                action = act_f[action]
                # Keep next_valid a Python list: the `if next_valid` mask below relies
                # on list truthiness (None / [] -> unmasked max).
                if next_valid:
                    next_valid = [act_f[a] for a in next_valid]
            if done:
                targets[i][action] = reward
            else:
                # Bootstrap only from actions that are legal in the next state; the network
                # is never trained on illegal actions, so their Q-values are junk. legal
                # None (== next_valid None/[]) means "no mask", matching the old behavior.
                legal = next_valid if next_valid else None
                if self.double_dqn:
                    online_row = online_next[i] if legal is None else online_next[i][legal]
                    pos = int(np.argmax(online_row))
                    a_star = pos if legal is None else legal[pos]
                    best_next = next_q[i][a_star]
                else:
                    best_next = np.max(next_q[i]) if legal is None else np.max(next_q[i][legal])
                targets[i][action] = reward + self.gamma * best_next
        self._train_step(tf.convert_to_tensor(states),
                          tf.convert_to_tensor(targets, dtype=tf.float32))

    def target_train(self):
        self.target_model.set_weights(self.model.get_weights())

    def polyak_update(self, tau):
        """Soft target update: target <- tau*online + (1-tau)*target. A small tau makes the
        target a slow exponential-moving-average of the online net (the smooth alternative
        to periodic hard copies; see train.py --target-sync-mode polyak)."""
        online_w = self.model.get_weights()
        target_w = self.target_model.get_weights()
        self.target_model.set_weights(
            [tau * o + (1.0 - tau) * t for o, t in zip(online_w, target_w)])

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
