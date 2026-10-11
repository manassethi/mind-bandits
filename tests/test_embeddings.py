import unittest
import numpy as np
from scripts.prepare_embeddings import embedding_feature_view,recover_cohort
from scripts.run_contextual_sweep import variants


class EmbeddingTests(unittest.TestCase):
    def test_view_excludes_tfidf_and_preserves_history(self):
        x=np.arange(2*3*12).reshape(2,3,12)
        result=embedding_feature_view(x,4,8)
        np.testing.assert_array_equal(result[:,:,:4],x[:,:,:4])
        np.testing.assert_array_equal(result[:,:,4:],x[:,:,8:])
        self.assertEqual(result.shape,(2,3,8))

    def test_sampling_replays_original_draws(self):
        train=[{} for _ in range(10)]
        dev=[{'user':str(i),'time':i,'history':[]} for i in range(8)]
        args={'seed':42,'full_data':False,'train_impressions':5,'dev_impressions':4,'users':3,'epochs':2}
        rng=np.random.default_rng(42)
        rng.choice(10,5,replace=False);rng.choice(8,4,replace=False)
        rng.permutation(5);rng.permutation(5)
        expected=[dev[i] for i in rng.choice(8,3,replace=False)]
        self.assertEqual(recover_cohort({'arguments':args},train,dev),expected)

    def test_embeddings_are_separate_policy_family(self):
        settings=variants([.1,.3],[.1],[.1],True,True)
        self.assertEqual(sum(f=='text' for _,f,_ in settings),2)
        self.assertEqual(sum(f=='embeddings' for _,f,_ in settings),2)


if __name__=='__main__':unittest.main()
