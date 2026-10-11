"""Bernoulli reward policies. Each policy sees only its chosen arm's reward."""
import numpy as np


class Policy:
    def __init__(self, arms, rng):
        self.rng = rng
        self.counts = np.zeros(arms, dtype=int)
        self.successes = np.zeros(arms)

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.successes[arm] += reward

    def means(self):
        return np.divide(self.successes, self.counts, out=np.zeros_like(self.successes), where=self.counts > 0)

    def best(self, scores):
        return int(self.rng.choice(np.flatnonzero(scores == scores.max())))


class ExploreThenCommit(Policy):
    def __init__(self, arms, rng, exploration_rounds):
        super().__init__(arms, rng)
        self.exploration_rounds = exploration_rounds
        self.committed = None

    def select(self):
        t = int(self.counts.sum())
        if t < self.exploration_rounds:
            return t % len(self.counts)
        if self.committed is None:
            self.committed = self.best(self.means())
        return self.committed


class EpsilonGreedy(Policy):
    def __init__(self, arms, rng, epsilon):
        super().__init__(arms, rng)
        self.epsilon = epsilon

    def select(self):
        unseen = np.flatnonzero(self.counts == 0)
        if len(unseen):
            return int(unseen[0])
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(self.counts)))
        return self.best(self.means())


class UCB1(Policy):
    def select(self):
        unseen = np.flatnonzero(self.counts == 0)
        if len(unseen):
            return int(unseen[0])
        return self.best(self.means() + np.sqrt(2 * np.log(self.counts.sum()) / self.counts))


class ThompsonSampling(Policy):
    def select(self):
        return self.best(self.rng.beta(1 + self.successes, 1 + self.counts - self.successes))
