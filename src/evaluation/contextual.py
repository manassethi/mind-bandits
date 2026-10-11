"""Shared round masks ensure cold-start comparisons use identical opportunities."""
import numpy as np


def phase_masks(candidates, new_articles, cold_round, cold_window):
    t = np.arange(len(candidates))
    offered = np.isin(candidates, new_articles).any(axis=1)
    window = (t >= cold_round) & (t < cold_round + cold_window)
    return {'all': np.ones(len(t), dtype=bool), 'before': t < cold_round,
            'after': t >= cold_round, 'cold_window': window,
            'new_offered': offered, 'cold_new_offered': offered & window}
