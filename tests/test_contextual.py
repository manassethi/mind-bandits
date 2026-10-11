import unittest
import numpy as np
from src.bandits.contextual import LinUCB, LinearThompson, AvailableBandit
from scripts.run_contextual import available_articles


class ContextualTests(unittest.TestCase):
    def test_inverse_update_matches_direct_ridge_solution(self):
        policy=LinUCB(3,np.random.default_rng(1))
        x=np.array([[1.,0.,.4],[1.,1.,.1],[1.,0.,.9]])
        y=np.array([1.,0.,1.])
        for feature,reward in zip(x,y):policy.update(feature,reward)
        np.testing.assert_allclose(policy.inverse,np.linalg.inv(np.eye(3)+x.T@x),atol=1e-12)
        np.testing.assert_allclose(policy.inverse@policy.b,np.linalg.solve(np.eye(3)+x.T@x,x.T@y),atol=1e-12)

    def test_new_articles_are_unavailable_before_boundary(self):
        np.testing.assert_array_equal(available_articles([4,2,8,7],2,9,10),[4,2])
        np.testing.assert_array_equal(available_articles([4,2,8,7],2,10,10),[4,2,8,7])

    def test_transfer_to_unseen_article_with_same_features(self):
        p=LinUCB(2,np.random.default_rng(3),scale=0)
        for _ in range(20):p.update(np.array([0.,1.]),1)
        self.assertEqual(p.select(np.array([[1.,0.],[0.,1.]])),1)

    def test_baselines_choose_only_offered_articles(self):
        for kind in ['Epsilon','UCB1','Thompson']:
            p=AvailableBandit(10,np.random.default_rng(1),kind)
            offered=np.array([3,7])
            for _ in range(50):
                choice=p.select(offered)
                self.assertIn(choice,[0,1]);p.update(offered[choice],1)
            self.assertEqual(p.counts.sum(),50)
            self.assertEqual(p.counts[[0,1,2,4,5,6,8,9]].sum(),0)

    def test_linear_thompson_is_reproducible(self):
        a=LinearThompson(2,np.random.default_rng(9))
        b=LinearThompson(2,np.random.default_rng(9))
        x=np.eye(2)
        for _ in range(30):
            chosen=a.select(x);self.assertEqual(chosen,b.select(x))
            a.update(x[chosen],1);b.update(x[chosen],1)


class ColdOpportunityTests(unittest.TestCase):
    def test_opportunity_mask_is_based_on_offered_articles(self):
        from src.evaluation.contextual import phase_masks
        candidates=np.array([[0,1],[1,0],[0,2],[1,0],[2,3],[3,1]])
        masks=phase_masks(candidates,[2,3],2,2)
        np.testing.assert_array_equal(masks['new_offered'],[False,False,True,False,True,True])
        np.testing.assert_array_equal(masks['cold_new_offered'],[False,False,True,False,False,False])

    def test_summary_averages_pools_before_intervals(self):
        from scripts.run_contextual_sweep import summarize
        rows=[]
        for seed,values in [(1,[0,10]),(2,[2,8])]:
            for pool,value in enumerate(values):
                rows.append(dict(seed=seed,pool=pool,policy='test',phase='all',clicks=value,ctr=value/10,regret=value,regret_per_round=value/10,evaluated_rounds=10))
        result=summarize(rows,['policy','phase'],True)[0]
        self.assertEqual(result['clicks_mean'],5)
        self.assertEqual(result['clicks_ci95_low'],5)
        self.assertEqual(result['clicks_ci95_high'],5)


if __name__=='__main__':unittest.main()
