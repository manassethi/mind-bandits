import unittest
import numpy as np
import torch
from src.features.sae import TopKSAE,preprocess_states
from scripts.analyze_sae import ablate_features
from scripts.run_contextual_sweep import variants


class SAETests(unittest.TestCase):
    def test_unit_state_control_removes_magnitude_and_handles_zero(self):
        states=np.array([[3.,4.],[30.,40.],[0.,0.]],dtype=np.float32)
        normalized=preprocess_states(states,'unit')
        np.testing.assert_allclose(normalized[0],normalized[1])
        np.testing.assert_allclose(np.linalg.norm(normalized,axis=1),[1,1,0])
        np.testing.assert_array_equal(states[0],[3,4])
        model=TopKSAE(2,8,2,np.zeros(2),1.)
        with torch.no_grad():codes=model.encode(torch.tensor(normalized)).numpy()
        np.testing.assert_allclose(codes[0],codes[1])
        self.assertIs(preprocess_states(states,'none'),states)
        with self.assertRaises(ValueError):preprocess_states(states,'invalid')

    def test_codes_are_nonnegative_and_top_k_sparse(self):
        torch.manual_seed(1)
        model=TopKSAE(4,12,3,np.zeros(4),1.)
        _,codes=model(torch.randn(20,4))
        self.assertTrue(bool((codes>=0).all()))
        self.assertTrue(bool(((codes>0).sum(axis=1)<=3).all()))

    def test_decoder_normalization_and_training_gradients(self):
        model=TopKSAE(4,12,3,np.zeros(4),1.)
        inputs=torch.randn(10,4);reconstructed,_=model(inputs)
        (reconstructed-inputs).square().mean().backward()
        self.assertIsNotNone(model.encoder.weight.grad)
        torch.testing.assert_close(model.decoder.weight.norm(dim=0),torch.ones(12))

    def test_ablation_removes_content_and_user_interaction_only(self):
        x=np.ones((2,8));changed=ablate_features(x,2,[1],3)
        np.testing.assert_array_equal(changed[:,[3,6]],np.zeros((2,2)))
        np.testing.assert_array_equal(changed[:,[0,1,2,4,5,7]],np.ones((2,6)))
        np.testing.assert_array_equal(x,np.ones((2,8)))

    def test_sae_is_separate_family(self):
        result=variants([.3],[.1],[.1],True,True,True)
        self.assertEqual(sum(f=='sae' for _,f,_ in result),1)


if __name__=='__main__':unittest.main()
