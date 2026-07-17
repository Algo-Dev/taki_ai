"""Tests for the DQN load path -- specifically the older-observation-contract adapter.

Self-contained on purpose: these build a synthetic old-contract checkpoint in a temp dir
rather than reaching for models/, which is gitignored, has been destroyed once already
(2026-07-12), and whose champions are not guaranteed to exist on any given machine.
"""
import shutil
import tempfile
import unittest

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from game import ACTION_SIZE, OBSERVATION_SIZE, Game
from agents.dqn import AIAgent
from agents.random import RandomAgent


def _save_net(path, obs_size):
    """A saved model on an arbitrary observation contract, shaped like create_model()."""
    m = keras.Sequential([layers.Dense(124, input_dim=obs_size, activation='relu'),
                          layers.Dense(64, activation='relu'),
                          layers.Dense(ACTION_SIZE)])
    m.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(learning_rate=0.001))
    m.save(path)


class ObsTruncationTest(unittest.TestCase):
    """A net from a SHORTER observation contract must still be playable for eval.

    Every observation change so far appended features, so today's vector is a strict superset
    of yesterday's and the leading prefix is exactly what the old net was trained on. Feeding
    it that prefix is faithful, not approximate -- which is what keeps an earlier champion
    (R6, 147 floats) measurable against a current net (150) instead of permanently
    uncomparable. Pinned because the day it silently breaks, cross-contract numbers keep
    printing and quietly mean nothing.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.old = f'{self.tmp}/old_net'
        _save_net(self.old, OBSERVATION_SIZE - 3)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_plays_on_the_leading_prefix_when_opted_in(self):
        a = AIAgent(epsilon=0.0, epsilon_min=0.0, load_model=self.old,
                    allow_obs_truncation=True)
        self.assertEqual(a.obs_size, OBSERVATION_SIZE - 3)
        g = Game([a, RandomAgent()], seed=1)
        g.reset()
        self.assertIsNotNone(a.play(g))   # would raise on a width mismatch

    def test_truncation_is_opt_in(self):
        # The default must refuse: an unnoticed truncation during a warm-start would train
        # an old-contract net while the appended features went nowhere.
        with self.assertRaises(RuntimeError) as cm:
            AIAgent(load_model=self.old)
        self.assertIn('allow_obs_truncation', str(cm.exception))

    def test_truncated_net_refuses_to_train(self):
        a = AIAgent(load_model=self.old, allow_obs_truncation=True)
        a.memory.extend([[np.zeros(OBSERVATION_SIZE), 0, 0.0, np.zeros(OBSERVATION_SIZE),
                          False, [0]]] * (a.batch_size + 1))
        with self.assertRaises(RuntimeError):
            a.replay()

    def test_longer_contract_is_rejected_outright(self):
        # A net wanting MORE floats than this build produces implies a REMOVED feature, so no
        # prefix of today's vector reconstructs its input. Truncation cannot rescue that, and
        # it must not pretend to.
        longer = f'{self.tmp}/longer_net'
        _save_net(longer, OBSERVATION_SIZE + 1)
        for flag in (False, True):
            with self.assertRaises(RuntimeError) as cm:
                AIAgent(load_model=longer, allow_obs_truncation=flag)
            self.assertIn('LONGER contract', str(cm.exception))

    def test_current_contract_net_is_untouched(self):
        current = f'{self.tmp}/current_net'
        _save_net(current, OBSERVATION_SIZE)
        a = AIAgent(load_model=current)
        self.assertEqual(a.obs_size, OBSERVATION_SIZE)   # no truncation, no opt-in needed


if __name__ == '__main__':
    unittest.main()
