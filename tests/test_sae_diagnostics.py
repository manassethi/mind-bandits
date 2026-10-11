import unittest
import numpy as np
from src.evaluation.sae_diagnostics import correlation,residualize,top_overlap,active_jaccard


class DiagnosticsTests(unittest.TestCase):
    def test_constant_feature_correlation_is_undefined(self):
        self.assertIsNone(correlation(np.ones(5),np.arange(5)))
        self.assertIsNone(correlation(np.ones(5),np.arange(5),True))

    def test_residualization_removes_linear_length_effect(self):
        controls=np.arange(10).reshape(-1,1)
        values=np.column_stack([2*controls[:,0]+3,-controls[:,0]+1])
        np.testing.assert_allclose(residualize(values,controls),0,atol=1e-12)

    def test_overlap_and_active_jaccard(self):
        self.assertEqual(top_overlap([5,4,0],[3,2,0],2),1)
        self.assertAlmostEqual(active_jaccard([1,1,0],[0,1,1]),1/3)
        self.assertIsNone(active_jaccard([0,0],[0,0]))


if __name__=='__main__':unittest.main()
