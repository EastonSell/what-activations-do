import unittest
import numpy as np
import core

class NumericalTests(unittest.TestCase):
    def setUp(self):
        self.p=core.initialize(101,3,4,3)
        self.x=np.array([[.2,.7,.1],[.9,.1,.6],[.3,.8,.4],[.6,.3,.2]])
        self.y=np.array([0,1,2,1])
    def test_forward_matches_independent_equations(self):
        h=np.clip(self.x.dot(self.p['W1'])+self.p['b1'],0,None)
        want=h.dot(self.p['W2'])+self.p['b2']
        np.testing.assert_array_equal(core.forward(self.p,self.x),want)
    def test_gradient_matches_central_differences(self):
        loss,g=core.loss_grad(self.p,self.x,self.y,.03,17)
        for key in self.p:
            for ix in np.ndindex(self.p[key].shape):
                old=self.p[key][ix]; eps=1e-6
                self.p[key][ix]=old+eps
                plus=core.loss_grad(self.p,self.x,self.y,.03,17)[0]
                self.p[key][ix]=old-eps
                minus=core.loss_grad(self.p,self.x,self.y,.03,17)[0]
                self.p[key][ix]=old
                self.assertAlmostEqual(g[key][ix],(plus-minus)/(2*eps),places=6)
    def test_adam_first_and_second_scalar_steps(self):
        p={'x':np.array([1.])}; state=core.Adam(p,.1,.9,.999,1e-8)
        state.step(p,{'x':np.array([2.])},('x',))
        self.assertAlmostEqual(p['x'][0],1-.1*2/(2+1e-8),places=12)
        state.step(p,{'x':np.array([2.])},('x',))
        self.assertAlmostEqual(p['x'][0],1-.2*2/(2+1e-8),places=12)
    def test_fixed_hidden_training_keeps_initial_bytes(self):
        cfg={'epochs':4,'batch_size':3,'learning_rate':.01,'alpha':.0001,'beta1':.9,'beta2':.999,'epsilon':1e-8}
        q,history=core.train(self.p,self.x,self.y,101,cfg,freeze_hidden=True)
        self.assertEqual(q['W1'].tobytes(),self.p['W1'].tobytes())
        self.assertEqual(q['b1'].tobytes(),self.p['b1'].tobytes())
        self.assertFalse(np.array_equal(q['W2'],self.p['W2']))
        self.assertEqual([r['updates'] for r in history],[2,4,6,8])
    def test_repeat_training_has_identical_parameters_and_schedule(self):
        cfg={'epochs':2,'batch_size':3,'learning_rate':.01,'alpha':.0001,'beta1':.9,'beta2':.999,'epsilon':1e-8}
        a,ha=core.train(self.p,self.x,self.y,211,cfg,False)
        b,hb=core.train(self.p,self.x,self.y,211,cfg,False)
        for key in a: np.testing.assert_array_equal(a[key],b[key])
        fields=('epoch','loss','updates','order_sha256')
        self.assertEqual([{k:r[k] for k in fields} for r in ha],[{k:r[k] for k in fields} for r in hb])
        self.assertEqual(sorted(core.epoch_order(9,211,0).tolist()),list(range(9)))
        self.assertFalse(np.array_equal(core.epoch_order(9,211,0),core.epoch_order(9,211,1)))
    def test_metrics_are_stable_and_use_strongest_foil(self):
        z=np.array([[1000.,999.,998.],[-1000.,-999.,-998.]])
        m=core.metrics(z,np.array([0,0]))
        np.testing.assert_allclose(m['margin'],[1.,-2.])
        np.testing.assert_array_equal(m['correct'],[True,False])
        np.testing.assert_allclose(m['ce'],[.407605964444,2.407605964444],atol=1e-11)
    def test_bad_shapes_and_nonfinite_inputs_are_rejected(self):
        for x in (np.ones((0,3)),np.array([[np.nan,0,0]]),np.ones(3)):
            with self.assertRaises(ValueError): core.forward(self.p,x)
        with self.assertRaises(ValueError): core.loss_grad(self.p,self.x,np.array([0,1,9,1]),0,4)

class InterventionTests(unittest.TestCase):
    def setUp(self):
        self.h=np.array([[2.,1.,3.],[4.,1.,3.],[6.,1.,3.],[8.,1.,3.]])
        self.mu,self.v,self.s=core.fit_geometry(self.h)
        self.q,self.r=core.random_bases(101,0,3)
    def test_centered_removal_preserves_mean_and_k0_is_identity(self):
        np.testing.assert_array_equal(core.intervene(self.h,self.mu,self.v,0,'pca'),self.h)
        out=core.intervene(self.h,self.mu,self.v,1,'pca')
        np.testing.assert_allclose(out,np.array([[5.,1.,3.]]*4),atol=1e-12)
    def test_full_rank_projection_is_constant_mean(self):
        for kind,basis in [('pca',self.v),('rank',self.q)]:
            out=core.intervene(self.h,self.mu,basis,3,kind)
            np.testing.assert_allclose(out,np.tile(self.mu,(4,1)),atol=1e-12)
    def test_random_energy_matches_each_example(self):
        top=core.intervene(self.h,self.mu,self.v,1,'pca')
        rnd=core.intervene(self.h,self.mu,self.v,1,'energy',self.r)
        np.testing.assert_allclose(np.linalg.norm(self.h-top,axis=1),np.linalg.norm(self.h-rnd,axis=1),atol=1e-12)
        self.assertFalse(np.allclose(top,rnd))
        np.testing.assert_allclose(self.q.T@self.q,np.eye(3),atol=1e-12)
        self.assertFalse(np.array_equal(self.q,self.r))
    def test_fit_is_train_only_not_affected_by_test_shift(self):
        mu,v,s=core.fit_geometry(self.h)
        core.intervene(self.h+100,mu,v,1,'pca')
        np.testing.assert_array_equal(mu,self.mu)
        np.testing.assert_array_equal(v,self.v)
    def test_gates_retain_failures_and_separate_comparability(self):
        self.assertEqual(core.gates([.95,.96],[.91,.92],.9,.05),{'primary_pass':True,'comparison_pass':True})
        self.assertEqual(core.gates([.95,.96],[.8,.92],.9,.05),{'primary_pass':True,'comparison_pass':False})
        self.assertEqual(core.gates([.89,.96],[.91,.92],.9,.05),{'primary_pass':False,'comparison_pass':False})
    def test_primary_statistic_preserves_example_pairing(self):
        a=np.array([[1.,3.],[3.,5.]])
        got=core.bootstrap_cases(a,100,1)
        np.testing.assert_array_equal(got['per_seed'],[2.,4.])
        self.assertEqual(got['mean'],3.)
        self.assertGreaterEqual(got['ci95'][0],2.)
        self.assertLessEqual(got['ci95'][1],4.)

if __name__=='__main__': unittest.main()

class TimingTests(unittest.TestCase):
    def test_each_epoch_records_finite_monotonic_elapsed_time(self):
        p=core.initialize(101,2,3,2)
        cfg={'epochs':3,'batch_size':2,'learning_rate':.01,'alpha':.0001,'beta1':.9,'beta2':.999,'epsilon':1e-8}
        _,history=core.train(p,np.array([[0.,1.],[1.,0.],[.5,.5]]),np.array([0,1,0]),101,cfg)
        prior=0.
        for row in history:
            self.assertIn('elapsed_seconds',row)
            self.assertIn('epoch_seconds',row)
            self.assertTrue(np.isfinite(row['elapsed_seconds']))
            self.assertGreaterEqual(row['elapsed_seconds'],prior)
            self.assertAlmostEqual(row['epoch_seconds'],row['elapsed_seconds']-prior,places=10)
            prior=row['elapsed_seconds']
