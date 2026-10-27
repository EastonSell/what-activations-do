import unittest
import numpy as np
from control import CorrelationDeletion, rescale

class Tests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[1,2,7],[2,5,7],[4,1,7],[8,7,7],[3,4,7.]],float)
        self.h = np.array([[6,3,9],[-1,4,2]],float)
        self.w = np.array([[2,-1],[3,4],[1,-2.]],float)
        self.b = np.array([2,-3.])

    def test_gauge_all_prescribed_scales_and_ranks(self):
        for s in ([1,1,1],[.25,4,2],[8,.5,.125]):
            for k in (0,1,2):
                with self.subTest(scale=s,rank=k):
                    xs, ws = rescale(self.x,self.w,s)
                    hs,_ = rescale(self.h,self.w,s)
                    a,b = CorrelationDeletion(self.x,k),CorrelationDeletion(xs,k)
                    np.testing.assert_allclose(a.projector,b.projector,rtol=0,atol=1e-10)
                    np.testing.assert_allclose(a.transform(self.h)*s,b.transform(hs),rtol=0,atol=1e-10)
                    np.testing.assert_allclose(a.transform(self.h)@self.w+self.b,b.transform(hs)@ws+self.b,rtol=0,atol=1e-10)
                    np.testing.assert_allclose(self.h@self.w+self.b,hs@ws+self.b,rtol=0,atol=1e-10)
    def test_constant_columns_preserved_even_off_training_support(self):
        for k in (0,1,2):
            np.testing.assert_array_equal(CorrelationDeletion(self.x,k).transform(self.h)[:,2],self.h[:,2])
    def test_all_constants(self):
        c=CorrelationDeletion(np.ones((4,3)),0)
        np.testing.assert_array_equal(c.transform(self.h),self.h)
    def test_tie_boundary_rejected(self):
        x=np.array([[-1,-1],[-1,1],[1,-1],[1,1.]])
        with self.assertRaisesRegex(ValueError,'tied'): CorrelationDeletion(x,1)
        CorrelationDeletion(x,0);CorrelationDeletion(x,2)
    def test_ordinary_pca_negative_control(self):
        x=np.array([[-1,-1],[-1,1],[1,-1],[1,1.]])+2
        w=np.array([[1,-1],[0,0.]])
        b=np.array([-2,2])
        accuracies=[]
        for s in ([4,1],[1,4]):
            xs,ws=rescale(x,w,s)
            z=xs-xs.mean(0)
            _,v=np.linalg.eigh(z.T@z/len(z)); p=np.eye(2)-np.outer(v[:,-1],v[:,-1])
            logits=(xs.mean(0)+z@p)@ws+b
            accuracies.append(np.mean(np.argmax(logits,1)==np.argmax(x@w+b,1)))
        self.assertEqual(accuracies,[.5,1.])
    def test_invalid_inputs(self):
        for x in ([],[1,2],[[np.nan]],[[np.inf]]):
            with self.assertRaises(ValueError): CorrelationDeletion(x,0)
        for k in (-1,3,True,1.5):
            with self.assertRaises(ValueError): CorrelationDeletion(self.x,k)
        for s in ([0,1,1],[-1,1,1],[1,np.inf,1],[1,2]):
            with self.assertRaises(ValueError):rescale(self.x,self.w,s)
        with self.assertRaises(ValueError):CorrelationDeletion(self.x,1).transform([[1,2]])
        with self.assertRaises(ValueError):CorrelationDeletion(self.x,1,float('nan'))
    def test_rescale_overflow_rejected(self):
        with self.assertRaisesRegex(ValueError,"unrepresentable"):
            rescale([[1e308]],[[1.]],[2.])
        with self.assertRaisesRegex(ValueError,"unrepresentable"):
            rescale([[1.]],[[1.]],[1e-320])
    def test_oblique_idempotent_not_symmetric(self):
        a=CorrelationDeletion(self.x,1)
        p=a.projector; d=a.scale
        t=p*d[None,:]/d[:,None]
        np.testing.assert_allclose(t@t,t,rtol=0,atol=1e-10)
        self.assertGreater(np.max(np.abs(t-t.T)),.01)

if __name__=='__main__':unittest.main(verbosity=2)
