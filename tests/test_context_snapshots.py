"""Regression checks for the CPU context aliasing found during final validation."""

import unittest

import numpy as np

from src.bandits.contextual import run


class ContextSnapshotTests(unittest.TestCase):
    def test_recorded_states_reconstruct_live_decisions(self):
        rng = np.random.default_rng(0)
        env = {"P": np.full((2, 4, 10), .04, np.float32), "n_live": 3,
               "horizon": 100, "lifetime": 60, "seed": 4}
        articles = rng.normal(size=(2, 10, 3)).astype(np.float32)
        users = rng.normal(size=(2, 4, 3)).astype(np.float32)
        ordinary = run(env, articles, users, "linucb_0.1", "cpu", seed=2, n_contexts=4)
        recorded = run(env, articles, users, "linucb_0.1", "cpu", seed=2, n_contexts=4, record_state=True)
        np.testing.assert_array_equal(ordinary["regret"], recorded["regret"])
        np.testing.assert_array_equal(recorded["ctx_live"][:, 0], [[0, 1, 2], [0, 1, 2]])
        self.assertFalse(np.array_equal(recorded["ctx_live"][:, 0], recorded["ctx_live"][:, -1]))
        for c in range(len(recorded["ctx_rounds"])):
            a = articles[np.arange(2)[:, None], recorded["ctx_live"][:, c]]
            u = users[np.arange(2), recorded["ctx_users"][:, c]][:, None, :]
            phi = np.concatenate([np.ones((2, 3, 1)), a, a * u], axis=2)
            mean = np.einsum("skd,sd->sk", phi, recorded["ctx_theta"][:, c])
            variance = np.einsum("skd,sde,ske->sk", phi, recorded["ctx_A_inv"][:, c], phi)
            scores = mean + .1 * np.sqrt(variance.clip(min=0))
            np.testing.assert_array_equal(scores.argmax(1), recorded["ctx_choice"][:, c])


if __name__ == "__main__":
    unittest.main()
