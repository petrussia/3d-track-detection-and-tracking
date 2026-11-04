import unittest

import numpy as np

try:
    from peleng.reference import select_reference, center_offset, reference_from_center
    from peleng.reference import orient_reference
except ModuleNotFoundError:
    select_reference = center_offset = reference_from_center = orient_reference = None
except ImportError:
    orient_reference = None


class ReferenceTests(unittest.TestCase):
    def test_rate_limited_axis_keeps_uncertainty_of_unresolved_motion_conflict(self):
        model=self.model()
        result=orient_reference(model,[-1000.,0.,0.],np.eye(2)*100.,.1)
        self.assertLessEqual(abs(result['yaw_rad']),.15000001)
        self.assertGreater(result['yaw_sigma_rad'],2.9)

    def test_fallback_axis_turns_from_reliable_body_motion_with_elapsed_time_bound(self):
        self.assertIsNotNone(orient_reference,'Body-motion reference orientation is missing')
        model=self.model()
        result=orient_reference(model,[0.,1000.,0.],np.eye(2)*100.,.2)
        self.assertGreater(result['front_cm'][1],200.)
        self.assertLessEqual(abs(result['yaw_rad']),.30000001)
        self.assertEqual(model['front_cm'],[2000.,0.,300.])

    def test_weak_or_stationary_body_velocity_cannot_rotate_front(self):
        self.assertIsNotNone(orient_reference,'Body-motion reference orientation is missing')
        model=self.model()
        for velocity in ([0.,0.,0.],[0.,100.,0.]):
            result=orient_reference(model,velocity,np.eye(2)*1e6,1.)
            self.assertEqual(result['front_cm'],model['front_cm'])
            self.assertGreater(result['yaw_sigma_rad'],model['yaw_sigma_rad'])

    def model(self):
        self.assertIsNotNone(select_reference, 'Physical reference mapping is missing')
        return {'position_cm':[1000.,0.,300.], 'center_cm':[1000.,0.,300.],
                'front_cm':[2000.,0.,300.], 'front_observable':True,
                'dimensions_cm':[2000.,400.,600.], 'yaw_rad':0., 'yaw_sigma_rad':.05,
                'covariance_cm2':np.diag([100.,100.,100.]).tolist(),
                'front_covariance_cm2':np.diag([400.,900.,100.]).tolist(),
                'method':'cuboid_multiview','cameras':[1,2]}

    def test_front_selection_uses_front_covariance_and_does_not_mutate_shape(self):
        model=self.model()
        model['systematic_covariance_cm2']=np.eye(3).tolist()
        model['front_systematic_covariance_cm2']=(np.eye(3)*400.).tolist()
        result=select_reference(model,'front')
        self.assertEqual(result['position_cm'],[2000.,0.,300.])
        self.assertEqual(result['covariance_cm2'][1][1],900.)
        self.assertEqual(result['reference_point'],'front_face_center')
        self.assertEqual(result['systematic_covariance_cm2'][0][0],400.)
        self.assertEqual(model['position_cm'],[1000.,0.,300.])

    def test_bearing_origin_is_shifted_together_with_reference_point(self):
        model=self.model()
        model['front_cm']=[1100.,200.,300.]
        base={'position_cm':[1300.,0.,300.], 'method':'bearing_update',
              'origin_cm':[0.,0.,300.], 'direction':[1.,0.,0.],
              'transverse_sigma_cm':100.,'covariance_cm2':np.diag([1e12,10000.,10000.]).tolist(),
              'metric_depth_observed':False,'cameras':[1]}
        result=reference_from_center(base,model,'front')
        self.assertEqual(result['origin_cm'],[100.,200.,300.])
        self.assertEqual(result['position_cm'],[1400.,200.,300.])
        self.assertEqual(result['center_cm'],[1300.,0.,300.])
        self.assertGreater(result['transverse_sigma_cm'],100.)
        self.assertGreater(np.trace(result['systematic_covariance_cm2']),0.)

    def test_ground_contact_proxy_uses_image_learned_height_without_fixed_offset(self):
        model=self.model()
        base={'position_cm':[1000.,0.,0.],'method':'ground_plane','cameras':[1],
              'covariance_cm2':np.diag([10000.,10000.,10000.]).tolist(), 'metric_depth_observed':False}
        result=reference_from_center(base,model,'front')
        self.assertEqual(result['center_cm'],[1000.,0.,300.])
        self.assertEqual(result['front_cm'],[2000.,0.,300.])
        self.assertEqual(result['method'],'ground_plane')
        self.assertFalse(result['metric_depth_observed'])
        self.assertEqual(center_offset(model,'front').tolist(),[1000.,0.,0.])

    def test_unresolved_forward_sign_never_invents_a_front(self):
        model=self.model()
        model.update(front_observable=False,front_cm=None)
        result=select_reference(model,'front')
        self.assertEqual(result['position_cm'],model['center_cm'])
        self.assertEqual(result['reference_point'],'cuboid_center')
