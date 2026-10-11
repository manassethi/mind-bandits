"""Shared linear reward models transfer evidence across articles via features."""
import numpy as np


class LinearPolicy:
    def __init__(self, dimensions, rng, scale=0.1, ridge=1.0):
        self.rng, self.scale = rng, scale
        self.inverse = np.eye(dimensions) / ridge
        self.b = np.zeros(dimensions)

    def update(self, feature, reward):
        projected = self.inverse @ feature
        self.inverse -= np.outer(projected, projected) / (1 + feature @ projected)
        self.inverse = (self.inverse + self.inverse.T) / 2
        self.b += reward * feature

    def best(self, scores):
        return int(self.rng.choice(np.flatnonzero(scores == scores.max())))


class LinUCB(LinearPolicy):
    def select(self, features):
        mean = features @ (self.inverse @ self.b)
        variance = np.einsum('ij,jk,ik->i', features, self.inverse, features)
        return self.best(mean + self.scale * np.sqrt(np.maximum(variance, 0)))


class LinearThompson(LinearPolicy):
    def select(self, features):
        # Gaussian linear posterior approximation for binary rewards.
        factor = np.linalg.cholesky(self.inverse)
        theta = self.inverse @ self.b + self.scale * factor @ self.rng.standard_normal(len(self.b))
        return self.best(features @ theta)


class AvailableBandit:
    def __init__(self, arms, rng, kind, epsilon=0.1):
        self.counts = np.zeros(arms, dtype=int)
        self.successes = np.zeros(arms)
        self.rng, self.kind, self.epsilon = rng, kind, epsilon

    def select(self, available):
        counts = self.counts[available]
        if self.kind == 'Thompson':
            scores = self.rng.beta(1+self.successes[available], 1+counts-self.successes[available])
        else:
            unseen = np.flatnonzero(counts == 0)
            if len(unseen): return int(self.rng.choice(unseen))
            if self.kind == 'Epsilon' and self.rng.random() < self.epsilon:
                return int(self.rng.integers(len(available)))
            scores = self.successes[available] / counts
            if self.kind == 'UCB1':
                scores += np.sqrt(2*np.log(max(self.counts.sum(), 1))/counts)
        return int(self.rng.choice(np.flatnonzero(scores == scores.max())))

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.successes[arm] += reward
