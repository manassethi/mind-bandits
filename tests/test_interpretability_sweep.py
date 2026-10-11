import unittest
from scripts.run_sae_interpretability_sweep import run_effects,effect_summary


class InterpretabilitySweepTests(unittest.TestCase):
    def test_top_random_contrast_has_correct_direction(self):
        rows=[dict(ablation='top_features',expected_ctr_change=-.02,choice_change_fraction=.3),
              dict(ablation='random_control_1',expected_ctr_change=-.005,choice_change_fraction=.1),
              dict(ablation='random_control_2',expected_ctr_change=.001,choice_change_fraction=.2)]
        result=run_effects(rows)
        self.assertAlmostEqual(result['random_ctr_drop'],.002)
        self.assertAlmostEqual(result['excess_ctr_drop'],.018)

    def test_intervals_use_seed_pool_averages(self):
        rows=[]
        for seed in [1,2,3]:
            for pool,value in enumerate([0.,.02]):
                rows.append(dict(seed=seed,pool=pool,top_ctr_drop=value,random_ctr_drop=0.,excess_ctr_drop=value,top_choice_change=value,random_choice_change=0.))
        results={r['metric']:r for r in effect_summary(rows)}
        self.assertEqual(results['top_ctr_drop']['mean'],.01)
        self.assertEqual(results['top_ctr_drop']['ci95_low'],.01)
        self.assertEqual(results['top_ctr_drop']['ci95_high'],.01)


if __name__=='__main__':unittest.main()
