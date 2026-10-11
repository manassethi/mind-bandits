import unittest
import numpy as np
from src.bandits.policies import ExploreThenCommit, UCB1, ThompsonSampling
from scripts.run_experiment import simulate
from src.evaluation.sweep import make_pools, interval


class ExperimentTests(unittest.TestCase):
    def test_pools_reproducible_without_duplicate_arms(self):
        arms = list(range(50))
        pools = make_pools(arms, 5, 5, 42)
        self.assertEqual(pools, make_pools(arms, 5, 5, 42))
        self.assertTrue(all(len(set(pool)) == 5 for pool in pools))
        self.assertGreater(len({tuple(pool) for pool in pools}), 1)

    def test_interval_detects_consistent_paired_gain(self):
        mean, low, high = interval([10, 11, 9, 10])
        self.assertEqual(mean, 10)
        self.assertGreater(low, 0)
        self.assertGreater(high, mean)

    def test_commit_is_fixed_after_exploration(self):
        p = ExploreThenCommit(2, np.random.default_rng(1), 4)
        for reward in [0, 1, 0, 1]:
            p.update(p.select(), reward)
        self.assertEqual(p.select(), 1)
        p.update(1, 0)
        self.assertEqual(p.select(), 1)

    def test_ucb_visits_every_arm(self):
        p = UCB1(3, np.random.default_rng(1))
        chosen = []
        for _ in range(3):
            chosen.append(p.select()); p.update(chosen[-1], 0)
        self.assertEqual(set(chosen), {0, 1, 2})

    def test_equal_probabilities_have_zero_regret(self):
        rng = np.random.default_rng(2)
        _, regret = simulate(ThompsonSampling(3, rng), np.ones(3) * 0.5, rng.random((100, 3)))
        np.testing.assert_array_equal(regret, np.zeros(100))

    def test_deterministic_rewards_and_regret(self):
        p = ExploreThenCommit(2, np.random.default_rng(1), 2)
        clicks, regret = simulate(p, np.array([0., 1.]), np.full((10, 2), 0.5))
        self.assertEqual(clicks[-1], 9)
        self.assertEqual(regret[-1], 1)
        self.assertEqual(p.counts.sum(), 10)


if __name__ == '__main__':
    unittest.main()
