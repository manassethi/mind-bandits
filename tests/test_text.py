import unittest
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from src.features.text import reward_matrix,text_policy_features
from scripts.run_contextual_sweep import variants


class TextTests(unittest.TestCase):
    def test_reward_features_do_not_depend_on_candidate_label(self):
        news={'a':dict(title='football team',abstract='',category='sports',subcategory='football'),
              'b':dict(title='football player',abstract='',category='sports',subcategory='football')}
        indices={'a':0,'b':1}
        lexical=HashingVectorizer(n_features=64,alternate_sign=False).transform([news[a]['title'] for a in indices])
        a,ya=reward_matrix([dict(history=['a'],impressions=[('b','0')])],news,indices,lexical)
        b,yb=reward_matrix([dict(history=['a'],impressions=[('b','1')])],news,indices,lexical)
        np.testing.assert_array_equal(a.toarray(),b.toarray())
        self.assertEqual(ya[0],0);self.assertEqual(yb[0],1)

    def test_text_features_preserve_category_history_block(self):
        base=np.ones((2,3,4))/3
        result=text_policy_features(base,np.eye(3),np.array([[1.,0.,0.],[0.,1.,0.]]))
        np.testing.assert_array_equal(result[:,:,:4],base)
        self.assertEqual(result.shape,(2,3,10))
        self.assertNotEqual(result[0,0,-3],result[1,0,-3])

    def test_text_variants_are_opt_in(self):
        self.assertFalse(any(f=='text' for _,f,_ in variants([.1],[.1],[.1])))
        self.assertEqual(sum(f=='text' for _,f,_ in variants([.1,.3],[.1],[.1],True)),2)


if __name__=='__main__':unittest.main()
