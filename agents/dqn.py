from collections import deque

import os

import numpy as np
import random

import tensorflow as tf
# This is a tiny network driven by single-sample / batch-64 ops, so TF's default
# thread pools (one op spread over every core) just thrash — system time and context
# switches dwarf the actual compute. A small pool is markedly faster here. Must be set
# before any TF op runs; harmless if the context is already initialised.
# TAKI_INTRA_OP/TAKI_INTER_OP shrink the pools further so a run can be squeezed onto a
# box that is already busy with other trainings (3 saturating TF processes wedge this
# WSL2 VM).
try:
    tf.config.threading.set_intra_op_parallelism_threads(
        int(os.environ.get('TAKI_INTRA_OP', '4')))
    tf.config.threading.set_inter_op_parallelism_threads(
        int(os.environ.get('TAKI_INTER_OP', '2')))
except RuntimeError:
    pass
from tensorflow import keras
from tensorflow.keras import layers

from game import (action_to_scalar, scalar_to_action, sym_tables,
                  OBSERVATION_SIZE, ACTION_SIZE)

# I would like to thank https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c
# for making an easy to read tutorial on DQN with Keras, I didn't know how to implement this and it really helped.


class AIAgent:

    def __init__(self, gamma=0.99, epsilon=1.0, epsilon_min=0.1, batch_size=64,
                 epsilon_decay=0.995, learning_rate=0.001, load_model=None,
                 color_sym=False, rank_sym=False, allow_obs_truncation=False, wide=False):
        super(AIAgent, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        # Network width. Default (narrow) is the historical lineage's 124->64 trunk that every
        # published champion trained on; wide is the 256->128->64 capacity-test arch (2.72x
        # params, A10's widening). Kept a flag rather than a hard default so the narrow lineage
        # stays reproducible from master -- M2 (RESEARCH_LOG 2026-07-18) showed wide buys ~no win
        # rate, so it is not a promotion, only an option. Loading is arch-agnostic either way:
        # a checkpoint whose width differs from this flag falls back to its own saved model.
        self.wide = wide
        # Replay-time symmetry augmentation: each sampled transition is trained under a
        # random relabeling of the four colors (color_sym, 24 perms) and/or of the nine
        # number ranks (rank_sym, 9! perms). Both are exact symmetries of TAKI's dynamics
        # and they compose; see sym_tables in game.py.
        self.color_sym = color_sym
        self.rank_sym = rank_sym
        self.epsilon_min = epsilon_min
        # Buffer sized for the all-seats training loop (A8): it collects ~4x more
        # transitions per trial (all four seats, not just the learner), so 80k keeps the
        # ~800-trial horizon that 20k gave under seat-0-only collection.
        self.memory = deque(maxlen=80000)
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        # Set by play() so a training loop can recover exactly which decision was made.
        self.last_state = None
        self.last_action = None
        self.model = self.create_model()
        self.target_model = self.create_model()
        # How many leading floats of the observation this net actually consumes. Equal to
        # OBSERVATION_SIZE for anything trained on the current contract; see _adopt_obs_size.
        self.obs_size = OBSERVATION_SIZE
        if load_model is not None:
            # A failed load must stop the process: silently falling back to random weights
            # turns an intended warm-start into a cold-start without anyone noticing.
            try:
                loaded = keras.models.load_model(load_model)
            except Exception as e:
                raise RuntimeError(f"could not load model from '{load_model}': {e}") from e
            try:
                self.model.set_weights(loaded.get_weights())
                self.target_model.set_weights(loaded.get_weights())
            except ValueError:
                # Shape mismatch against today's create_model() -- this checkpoint predates
                # an architecture change (see CLAUDE.md). Can't warm-start training across
                # that boundary, but for eval (epsilon=0, no replay/optimizer use) it's fine
                # to just adopt the checkpoint's own saved architecture directly, so old and
                # new snapshots can still be compared/played against each other in-process.
                self.model = loaded
                self.target_model = keras.models.load_model(load_model)
                self._adopt_obs_size(load_model, allow_obs_truncation)

    def _adopt_obs_size(self, load_model, allow_obs_truncation):
        """Let a net from an older, SHORTER observation contract still play.

        Every observation change so far has APPENDED features, so today's vector is a strict
        superset of yesterday's: obs[:N] is bit-identical to what an N-float net was trained on
        for every past contract (147 for R6, 150 for the seat-count-one-hot nets, 162 since the
        R4 color-void block; pinned by gametest). Feeding such a net the leading prefix is
        exact, not an approximation -- it sees precisely its own observation, minus only the
        features it never had. That is what lets a pre-one-hot champion like R6 sit at the
        same table as a current net instead of being permanently uncomparable.

        Opt-in and eval-only. Truncating during TRAINING would quietly train an old-contract
        net while the new features went nowhere, so replay() refuses (see below) and
        train.py never passes the flag.
        """
        dim = self.model.layers[0].input_shape[-1]
        if dim == OBSERVATION_SIZE:
            return
        if dim > OBSERVATION_SIZE:
            raise RuntimeError(
                f"'{load_model}' expects {dim} observation floats but this build produces only "
                f"{OBSERVATION_SIZE}. That checkpoint is from a LONGER contract (a removed "
                f"feature, not an appended one), so no prefix of today's vector reconstructs "
                f"it. It cannot be played here.")
        if not allow_obs_truncation:
            raise RuntimeError(
                f"'{load_model}' expects {dim} observation floats, this build produces "
                f"{OBSERVATION_SIZE}. It predates an observation change. Pass "
                f"allow_obs_truncation=True to play it on the leading {dim} floats (exact -- "
                f"the observation only ever grew by appending); this is for EVAL only, and "
                f"warm-starting training across the boundary is not supported.")
        self.obs_size = dim
        print(f'NOTE: {load_model} is on an older observation contract ({dim} floats vs '
              f'{OBSERVATION_SIZE}); playing it on the leading {dim}. It cannot see the '
              f'{OBSERVATION_SIZE - dim} appended feature(s).')

    def create_model(self):
        model = keras.Sequential()
        if self.wide:
            # Capacity-test arch (the MIXED-COUNT question, M2): the net must represent three
            # different games (2/3/4 seats), not one, and M1 sat ~1 SE below the per-count best
            # at every count -- the signature of multi-task interference. This is A10's exact
            # widening (124->64 -> 256->128->64, +one layer, 2.72x params) so the result is
            # comparable; A10 found it inert at four seats SINGLE-COUNT, but that says nothing
            # about the multi-task union. Input/output sizes are unchanged, so a checkpoint of
            # either width still plays via the cross-architecture eval fallback.
            model.add(layers.Dense(256, input_dim=OBSERVATION_SIZE, activation="relu"))
            model.add(layers.Dense(128, activation="relu"))
            model.add(layers.Dense(64, activation="relu"))
        else:
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
        # A truncated net is an eval adapter, not a trainable model: the symmetry tables and
        # the observations in the buffer are both full-width, and training here would fit an
        # old-contract net while the appended features silently went nowhere.
        if self.obs_size != OBSERVATION_SIZE:
            raise RuntimeError(
                f'refusing to train a net on the older {self.obs_size}-float observation '
                f'contract (this build produces {OBSERVATION_SIZE}); obs truncation is for '
                f'eval only.')
        if len(self.memory) < self.batch_size:
            return
        samples = random.sample(self.memory, self.batch_size)
        states = np.array([s[0] for s in samples], dtype=np.float32)       # (batch, OBS)
        next_states = np.array([s[3] for s in samples], dtype=np.float32)  # (batch, OBS)
        augment = self.color_sym or self.rank_sym
        if augment:
            # One uniformly-random relabeling per transition (identity included); state,
            # new_state, action and next_valid all get the SAME permutation, so each row
            # stays a genuine environment transition (reward/done depend only on hand sizes,
            # so they are invariant under both relabelings). Must happen BEFORE the model()
            # calls below: the target rows are the predictions on the augmented states, so
            # the untouched entries keep target == prediction (zero gradient). The buffer
            # keeps the originals — states/next_states are fresh copies and take_along_axis
            # allocates new arrays.
            obs_gather, act_fwd = sym_tables(self.batch_size, self.color_sym, self.rank_sym)
            states = np.take_along_axis(states, obs_gather, axis=1)
            next_states = np.take_along_axis(next_states, obs_gather, axis=1)
        # Direct model() calls instead of model.predict() — far less per-call overhead
        # for batches this small.
        targets = self.model(states, training=False).numpy()       # (batch, ACTION_SIZE)
        next_q = self.target_model(next_states, training=False).numpy()
        for i, (_, action, reward, _, done, next_valid) in enumerate(samples):
            if augment:
                act_f = act_fwd[i]
                action = act_f[action]
                # Keep next_valid a Python list: the `if next_valid` mask below relies
                # on list truthiness (None / [] -> unmasked max).
                if next_valid:
                    next_valid = [act_f[a] for a in next_valid]
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
        # The slice is a no-op (obs_size == OBSERVATION_SIZE) for everything except a net
        # loaded from an older, shorter contract; see _adopt_obs_size.
        x = state[np.newaxis, :self.obs_size].astype(np.float32)
        q_values = self.model(x, training=False).numpy()[0]      # (ACTION_SIZE,)
        return actions[int(np.argmax(q_values[actions]))]

    def play(self, game):
        state = game.observation()
        actions = list(map(lambda x: action_to_scalar(*x), game.valid_moves()))
        action = self.act(state, actions)
        self.last_state = state
        self.last_action = action
        return scalar_to_action(action)
