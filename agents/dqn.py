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


class DuelingAggregation(layers.Layer):
    """Dueling recombination Q = V + (A - mean_legal(A)).

    The mean is taken over the LEGAL actions only (the second input is a 0/1 legality
    mask over the 65 action slots), not all 65: most actions are illegal in any given
    state, so centring by the full-vector mean would fold untrained illegal-action
    advantages into every legal Q. Q-values of illegal actions are still produced but
    are junk and never selected (act() and the bootstrap both mask)."""

    def call(self, inputs):
        V, A, mask = inputs
        # A row's mask is never all-zeros in practice (cur states always have a legal
        # move; empty next-state masks are made all-ones by replay() since those rows
        # are discarded), but guard the division so a stray empty mask can't NaN.
        legal_count = tf.maximum(tf.reduce_sum(mask, axis=1, keepdims=True), 1.0)
        a_mean = tf.reduce_sum(A * mask, axis=1, keepdims=True) / legal_count
        return V + (A - a_mean)


class AIAgent:

    def __init__(self, gamma=0.99, epsilon=1.0, epsilon_min=0.1, batch_size=64,
                 epsilon_decay=0.995, learning_rate=0.001, load_model=None,
                 color_sym=False):
        super(AIAgent, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        # Replay-time color-symmetry augmentation: TAKI's colors are interchangeable,
        # so each sampled transition is trained under a random relabeling of the four
        # colors (see the COLOR_PERMS tables in game.py).
        self.color_sym = color_sym
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
        self.last_valid = None
        self.model = self.create_model()
        self.target_model = self.create_model()
        if load_model is not None:
            # A failed load must stop the process: silently falling back to random weights
            # turns an intended warm-start into a cold-start without anyone noticing.
            custom = {"DuelingAggregation": DuelingAggregation}
            try:
                loaded = keras.models.load_model(load_model, custom_objects=custom)
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
                self.target_model = keras.models.load_model(load_model, custom_objects=custom)
        # Whether self.model is the two-input dueling net (obs + legality mask) or a legacy
        # single-input net adopted via the fallback above. act() must call each with the
        # input signature it expects, so a dueling agent can still play/eval against old
        # flat checkpoints (the fallback path) and vice-versa.
        self.dueling = len(self.model.inputs) == 2

    def create_model(self):
        # Dueling head: the 124->64 trunk feeds a scalar state-value V(s) and a per-action
        # advantage A(s,a), recombined as Q = V + (A - mean_legal(A)). The heads split
        # directly off the 64-wide trunk with no per-stream hidden layer, so capacity stays
        # ~equal to the old flat net (+65 params for V) -- this isolates the dueling
        # parametrization from width, which was tested separately (A10) and did not help.
        # A second input carries the legality mask needed for the masked mean.
        obs_in = keras.Input(shape=(OBSERVATION_SIZE,))
        mask_in = keras.Input(shape=(ACTION_SIZE,))            # 1.0 = legal, 0.0 = illegal
        h = layers.Dense(124, activation="relu")(obs_in)
        h = layers.Dense(64, activation="relu")(h)
        value = layers.Dense(1)(h)                             # (batch, 1)
        advantage = layers.Dense(ACTION_SIZE)(h)               # (batch, 65)
        q = DuelingAggregation()([value, advantage, mask_in])  # (batch, 65)
        model = keras.Model([obs_in, mask_in], q)
        model.compile(loss="mean_squared_error",
                      optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, new_state, done, next_valid=None,
                 cur_valid=None):
        # next_valid: the action scalars that are legal in new_state, used to mask the
        # bootstrap target so it never relies on the Q-value of an illegal action.
        # cur_valid: the action scalars that are legal in state, needed to build the
        # legality mask the dueling head consumes on the forward pass over `state`.
        self.memory.append([state, action, reward, new_state, done, next_valid,
                            cur_valid])

    @tf.function
    def _train_step(self, states, masks, targets):
        # Compiled single gradient step (replaces model.fit, which retraces per call in
        # eager and carries heavy Python overhead). MSE over the full action vector — the
        # non-taken actions have target == current prediction, so they contribute no
        # gradient, exactly as the previous fit-based update did. `masks` are the same
        # current-state legality masks used to seed `targets`, so predictions match.
        with tf.GradientTape() as tape:
            preds = self.model([states, masks], training=True)
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
        if self.color_sym:
            # One uniformly-random color relabeling per transition (identity included);
            # state, new_state, action and the valid-action lists all get the SAME
            # permutation, so each row stays a genuine environment transition (reward/done
            # are color-invariant). Must happen BEFORE the model() calls below: the target
            # rows are the predictions on the augmented states, so the 63 untouched
            # entries keep target == prediction (zero gradient). The buffer keeps the
            # originals — states/next_states are fresh copies and take_along_axis
            # allocates new arrays.
            ks = np.random.randint(len(OBS_PERMS), size=self.batch_size)
            states = np.take_along_axis(states, OBS_PERMS[ks], axis=1)
            next_states = np.take_along_axis(next_states, OBS_PERMS[ks], axis=1)
        # Remap actions/valid-lists under the same permutation, then build the legality
        # masks the dueling head needs. This must precede the forward passes, so it can't
        # be folded into the target-assignment loop below (which runs after them). The
        # masks centre the advantage stream over legal actions only (see
        # DuelingAggregation); each row's mask must line up with its (permuted) obs.
        actions = [s[1] for s in samples]
        next_valids = [s[5] for s in samples]
        cur_valids = [s[6] for s in samples]
        cur_masks = np.zeros((self.batch_size, ACTION_SIZE), dtype=np.float32)
        next_masks = np.zeros((self.batch_size, ACTION_SIZE), dtype=np.float32)
        for i in range(self.batch_size):
            if self.color_sym:
                act_f = ACT_PERMS[ks[i]]
                actions[i] = act_f[actions[i]]
                # Keep the valid lists as Python lists: the `if next_valid` mask in the
                # loop below relies on list truthiness (None / [] -> unmasked max).
                if next_valids[i]:
                    next_valids[i] = [act_f[a] for a in next_valids[i]]
                if cur_valids[i]:
                    cur_valids[i] = [act_f[a] for a in cur_valids[i]]
            cur_masks[i, cur_valids[i]] = 1.0
            # Terminal / fallback rows have no next-legal set; their next_q is discarded,
            # but give them an all-ones mask so DuelingAggregation never divides by zero.
            if next_valids[i]:
                next_masks[i, next_valids[i]] = 1.0
            else:
                next_masks[i, :] = 1.0
        # Direct model() calls instead of model.predict() — far less per-call overhead
        # for batches this small.
        targets = self.model([states, cur_masks], training=False).numpy()       # (batch, ACTION_SIZE)
        next_q = self.target_model([next_states, next_masks], training=False).numpy()
        for i, (_, _, reward, _, done, _, _) in enumerate(samples):
            action = actions[i]
            next_valid = next_valids[i]
            if done:
                targets[i][action] = reward
            else:
                # Bootstrap only from actions that are legal in the next state; the
                # network is never trained on illegal actions, so their Q-values are junk.
                best_next = np.max(next_q[i][next_valid]) if next_valid else np.max(next_q[i])
                targets[i][action] = reward + self.gamma * best_next
        self._train_step(tf.convert_to_tensor(states),
                          tf.convert_to_tensor(cur_masks),
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
        if self.dueling:
            mask = np.zeros((1, ACTION_SIZE), dtype=np.float32)  # legality mask for the dueling head
            mask[0, actions] = 1.0
            q_values = self.model([x, mask], training=False).numpy()[0]  # (ACTION_SIZE,)
        else:
            q_values = self.model(x, training=False).numpy()[0]          # legacy single-input net
        return actions[int(np.argmax(q_values[actions]))]

    def play(self, game):
        state = game.observation()
        actions = list(map(lambda x: action_to_scalar(*x), game.valid_moves()))
        action = self.act(state, actions)
        self.last_state = state
        self.last_action = action
        # Legal action scalars for `state`, so the training loop can persist the mask the
        # dueling head needs on the forward pass over this state.
        self.last_valid = actions
        return scalar_to_action(action)
