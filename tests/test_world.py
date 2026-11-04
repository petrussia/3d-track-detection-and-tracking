import math
import unittest

import numpy as np

try:
    from peleng.world import WorldTrack
except ModuleNotFoundError:
    WorldTrack = None


def fix(x, y=0, z=100, sigma=50):
    return {'position_cm': [x, y, z], 'method': 'triangulated', 'cameras': [1, 2],
            'covariance_cm2': (np.eye(3) * sigma**2).tolist()}


class WorldTests(unittest.TestCase):
    def test_compatible_metric_after_short_gap_uses_gates_without_reset(self):
        track = self.make()
        track.update(fix(1000), 0.)
        track.update(None, .5)
        single = fix(1020)
        single.update(method='cuboid_singleview', metric_depth_observed=False)
        result = track.update(single, .6)
        self.assertTrue(result['measurement_accepted'])
        self.assertFalse(result['mode_reset'])
        self.assertEqual(result['segment_id'], 0)
        self.assertIn('track_expired', result['transition_reasons'])

    def test_translated_bearing_retains_growing_common_reference_uncertainty(self):
        track=self.make()
        track.update(fix(1000),0)
        bearing=fix(1000)
        bearing.update(method='bearing_update',origin_cm=[0.,0.,100.],direction=[1.,0.,0.],
                       transverse_sigma_cm=50.,systematic_covariance_cm2=np.diag([0.,90000.,0.]).tolist())
        result=track.update(bearing,.1)
        self.assertTrue(result['measurement_accepted'])
        self.assertGreaterEqual(result['uncertainty_m'],3.)

    def test_sustained_thirty_fps_multiview_poses_can_reacquire(self):
        track=self.make()
        track.update(fix(0),0)
        bearing=fix(0)
        bearing.update(method='bearing_update',origin_cm=[-1000.,0.,100.],
                       direction=[1.,0.,0.],transverse_sigma_cm=50.)
        for frame in range(1,6):
            pose=fix(1500+frame*10)
            pose.update(method='cuboid_multiview',fit_normalized_rms=.5,
                        metric_depth_observed=True,scale_usable=True,partial_silhouette=False)
            result=track.update(pose,frame/30)
            if result['measurement_accepted']:
                self.assertEqual(result['reacquisition_reason'],'consistent_multiview_pose')
                break
            track.update(bearing,frame/30+.005)
        else:
            self.fail('A valid 30 FPS metric stream could never reset the biased depth')

    def test_consistent_full_multiview_reacquires_while_bearings_keep_old_depth_alive(self):
        track=self.make()
        track.update(fix(0),0)
        bearing=fix(0)
        bearing.update(method='bearing_update', origin_cm=[-1000.,0.,100.],
                       direction=[1.,0.,0.], transverse_sigma_cm=50.)
        last=None
        for timestamp,x in [(.1,1500),(.2,1510),(.3,1520)]:
            pose=fix(x)
            pose.update(method='cuboid_multiview',fit_normalized_rms=.5,
                        metric_depth_observed=True,scale_usable=True,partial_silhouette=False)
            last=track.update(pose,timestamp)
            if timestamp<.3:
                self.assertFalse(last['measurement_accepted'])
                track.update(bearing,timestamp+.03)
        self.assertTrue(last['measurement_accepted'])
        self.assertEqual(last['reacquisition_reason'],'consistent_multiview_pose')
        self.assertAlmostEqual(last['position_cm'][0],1520)
        self.assertTrue(last['reinitialized'])

    def test_geometric_ray_agreement_alone_cannot_force_depth_reacquisition(self):
        track=self.make()
        track.update(fix(0),0)
        bearing=fix(0)
        bearing.update(method='bearing_update',origin_cm=[-1000.,0.,100.],
                       direction=[1.,0.,0.],transverse_sigma_cm=50.)
        for timestamp in [.1,.2,.3,.4]:
            raw=fix(1500)
            raw['residual_cm']=.01
            result=track.update(raw,timestamp)
            self.assertFalse(result['measurement_accepted'])
            track.update(bearing,timestamp+.03)

    def test_reference_change_requires_confirmation_instead_of_instant_reset(self):
        track = self.make()
        center = fix(0)
        center['reference_point'] = 'bbox_visual_center'
        a = track.update(center, 0)
        front = fix(1000)
        front.update(method='cuboid_multiview', reference_point='front_face_center',
                     metric_depth_observed=True, scale_usable=True, fit_normalized_rms=.5)
        b = track.update(front, .1)
        self.assertFalse(b['measurement_accepted'])
        self.assertIn('reference_changed', b['transition_reasons'])
        track.update(front, .2)
        b = track.update(front, .3)
        self.assertTrue(b['measurement_accepted'])
        self.assertEqual(b['velocity_cm_s'], [0.,0.,0.])
        self.assertGreater(b['segment_id'], a['segment_id'])
        self.assertEqual(b['reference_point'], 'front_face_center')
        self.assertTrue(b['metric_depth_observed'])

    def make(self, **kwargs):
        self.assertIsNotNone(WorldTrack, 'causal world filter is missing')
        return WorldTrack(**kwargs)

    def test_outlier_does_not_teleport_and_preserves_raw_evidence(self):
        t = self.make()
        t.update(fix(0), 0)
        t.update(fix(100), .1)
        result = t.update(fix(100000), .2)
        self.assertEqual(result['state'], 'coasting')
        self.assertFalse(result['measurement_accepted'])
        self.assertLess(abs(result['position_cm'][0]), 500)
        self.assertEqual(result['raw_measurement']['position_cm'][0], 100000)
        self.assertTrue(result['rejection_reason'])

    def test_predicts_motion_with_variable_dt_then_expires(self):
        t = self.make(max_coast_s=.5)
        for i in range(11):
            t.update(fix(i * 100), i / 10)
        result = t.update(None, 1.2)
        self.assertEqual(result['method'], 'prediction')
        self.assertGreater(result['position_cm'][0], 1100)
        self.assertLess(result['position_cm'][0], 1300)
        self.assertEqual(t.update(None, 1.6)['state'], 'lost')
        self.assertIsNone(t.update(None, 1.7)['position_cm'])

    def test_prediction_uncertainty_grows_and_long_loss_requires_confirmed_pose(self):
        t = self.make()
        first = t.update(fix(100), 0)
        coast = t.update(None, .3)
        self.assertGreater(coast['uncertainty_m'], first['uncertainty_m'])
        t.update(None, 2)
        reacquired = t.update(fix(9000), 2.1)
        self.assertEqual(reacquired['state'], 'lost')
        pose = fix(9000)
        pose.update(method='cuboid_multiview', metric_depth_observed=True,
                    scale_usable=True, fit_normalized_rms=.5)
        t.update(pose, 2.2)
        t.update(pose, 2.3)
        reacquired = t.update(pose, 2.4)
        self.assertEqual(reacquired['state'], 'observed')
        self.assertAlmostEqual(reacquired['position_cm'][0], 9000)
        self.assertTrue(reacquired['reinitialized'])

    def test_bearing_only_updates_do_not_renew_metric_depth_indefinitely(self):
        t = self.make(max_bearing_s=1.0)
        t.update(fix(1000), 0)
        bearing = fix(1000)
        bearing['method'] = 'bearing_update'
        bearing['covariance_cm2'] = np.diag([1e10, 2500, 2500]).tolist()
        bearing.update(origin_cm=[0, 0, 100], direction=[1., 0., 0.], transverse_sigma_cm=50.)
        self.assertEqual(t.update(bearing, .5)['state'], 'observed')
        result = t.update(bearing, 1.1)
        self.assertEqual(result['state'], 'lost')
        self.assertIsNone(result['position_cm'])

    def test_nonfinite_backward_times_and_bad_covariance_are_rejected(self):
        t = self.make()
        t.update(fix(0), 1)
        for timestamp in [math.nan, math.inf, .5]:
            with self.assertRaises(ValueError):
                t.update(None, timestamp)
        bad = fix(0)
        bad['covariance_cm2'][0][0] = -1
        with self.assertRaises(ValueError):
            t.update(bad, 1.1)

    def test_identical_rays_do_not_collapse_along_ray_uncertainty(self):
        t = self.make(max_bearing_s=10)
        t.update(fix(1000, z=200, sigma=1000), 0)
        before = t.p[0, 0]
        ray = {'position_cm': [1000, 0, 200], 'method': 'bearing_update',
               'origin_cm': [0, 0, 200], 'direction': [1., 0., 0.],
               'transverse_sigma_cm': 50., 'covariance_cm2': np.diag([1e12, 2500, 2500]).tolist()}
        for i in range(1, 101):
            result = t.update(ray, i*.01)
        self.assertGreaterEqual(t.p[0, 0], before)
        self.assertAlmostEqual(result['position_cm'][2], 200)

    def test_incomplete_bearing_never_falls_back_to_a_metric_fix(self):
        t = self.make()
        t.update(fix(1000), 0)
        bad = fix(1000)
        bad['method'] = 'bearing_update'
        with self.assertRaises(ValueError):
            t.update(bad, .1)
        bad.update(origin_cm=[0, 0, 100], direction=[1., 0., 0.], transverse_sigma_cm=-5.)
        with self.assertRaises(ValueError):
            t.update(bad, .2)

    def test_bearing_posterior_behind_camera_is_rejected_before_commit(self):
        import tempfile
        from test_geometry import write_camera
        from peleng.geometry import CameraRig, estimate_measurement
        with tempfile.TemporaryDirectory() as folder:
            write_camera(folder, 1, [(0, 0, 0, 100, 0, 0, 0)])
            write_camera(folder, 2, [(0, 0, 1000, 100, 0, 0, 0)])
            rig = CameraRig(folder, width=400, height=200)
            t = self.make()
            initial = estimate_measurement(rig, 0, [{'camera_id': 1, 'bbox': [200,90,220,110], 'confidence': .9}])
            # A deliberately uncertain metric prior exercises posterior cheirality;
            # ordinary ground mode never permits unconstrained bearing updates.
            initial['method'] = 'triangulated'
            initial['covariance_cm2'] = (np.asarray(initial['covariance_cm2']) + np.asarray(initial['systematic_covariance_cm2'])).tolist()
            t.update(initial, 0)
            bearing = estimate_measurement(rig, 0, [{'camera_id': 2, 'bbox': [210,90,230,110], 'confidence': .9}],
                                           prior_cm=t.predict_position(.1), allow_ground=False)
            result = t.update(bearing, .1)
            self.assertFalse(result['measurement_accepted'])
            self.assertEqual(result['rejection_reason'], 'behind_camera')
            self.assertGreater(result['position_cm'][0], 0)

    def test_ground_prior_is_a_separate_mode_and_triangulation_resets_velocity(self):
        t = self.make()
        ground = fix(1000, z=0)
        ground['method'] = 'ground_plane'
        first = t.update(ground, 0)
        active = getattr(t, 'depth_prior_active', None)
        self.assertIsNotNone(active, 'explicit metric-depth mode is missing')
        self.assertFalse(active(.1))
        ground['position_cm'][0] = 1100
        t.update(ground, .1)
        result = t.update(fix(4000, z=300), .2)
        self.assertFalse(result['measurement_accepted'])
        t.update(fix(4000, z=300), .3)
        result = t.update(fix(4000, z=300), .4)
        self.assertEqual(result['mode'], 'metric_3d')
        self.assertEqual(result['velocity_cm_s'], [0., 0., 0.])
        self.assertEqual(result['position_cm'], [4000., 0., 300.])
        self.assertGreater(result['segment_id'], first['segment_id'])
        self.assertTrue(active(.3))
        self.assertFalse(active(6))

    def test_expired_singleview_cannot_relocate_even_when_repeated(self):
        t = self.make()
        t.update(fix(1000), 0)
        single = fix(9000)
        single.update(method='cuboid_singleview', metric_depth_observed=False)
        for timestamp in (.533, .6, .7, .8):
            r = t.update(single, timestamp)
            self.assertFalse(r['measurement_accepted'])
            self.assertFalse(r['mode_reset'])
            self.assertIsNone(r['position_cm'])
            self.assertIn('track_expired', r['transition_reasons'])
            self.assertAlmostEqual(r['predicted_position_cm'][0], 1000)

    def test_incoherent_multiview_candidates_do_not_confirm(self):
        t = self.make()
        t.update(fix(0), 0)
        for timestamp, x in ((1., 10000), (1.1, -10000), (1.2, 10000)):
            pose = fix(x)
            pose.update(method='cuboid_multiview', metric_depth_observed=True,
                        scale_usable=True, fit_normalized_rms=.5)
            r = t.update(pose, timestamp)
            self.assertFalse(r['measurement_accepted'])
            self.assertFalse(r['mode_reset'])

    def test_broad_uncertainty_does_not_allow_instant_hundred_metre_correction(self):
        t = self.make()
        t.update(fix(0, sigma=100000), 0)
        result = t.update(fix(50000), .1)
        self.assertFalse(result['measurement_accepted'])
        self.assertEqual(result['rejection_reason'], 'motion_gate')
        self.assertLess(abs(result['position_cm'][0]), 1000)

    def test_ground_camera_handoff_is_gated_without_reset(self):
        t = self.make()
        first = fix(1000, z=0)
        first.update(method='ground_plane', cameras=[1])
        a = t.update(first, 0)
        second = fix(15000, z=0)
        second.update(method='ground_plane', cameras=[2])
        b = t.update(second, .1)
        self.assertEqual(b['velocity_cm_s'], [0., 0., 0.])
        self.assertFalse(b['measurement_accepted'])
        self.assertFalse(b['mode_reset'])
        self.assertEqual(b['segment_id'], a['segment_id'])
        self.assertAlmostEqual(b['position_cm'][0],1000)

    def test_singleview_range_support_does_not_renew_independent_depth(self):
        t=self.make(max_bearing_s=1.)
        t.update(fix(1000),0)
        single=fix(1000)
        single.update(method='cuboid_singleview',metric_depth_observed=False)
        for timestamp in [.2,.4,.6,.8,1.1]:
            result=t.update(single,timestamp)
            self.assertTrue(result['measurement_accepted'])
            self.assertAlmostEqual(result['depth_support_age_s'],0.)
            self.assertAlmostEqual(result['independent_depth_age_s'],timestamp)
        self.assertEqual(t.update(fix(1000),1.2)['independent_depth_age_s'],0.)

    def test_partial_singleview_does_not_move_track_even_with_perfect_pixel_fit(self):
        t=self.make()
        t.update(fix(1000),0)
        partial=fix(1400,z=-300)
        partial.update(method='cuboid_singleview',metric_depth_observed=False,
                       partial_silhouette=True,fit_rms_px=0.)
        result=t.update(partial,.1)
        self.assertFalse(result['measurement_accepted'])
        self.assertEqual(result['rejection_reason'],'partial_singleview_depth')
        self.assertEqual(result['position_cm'],[1000.,0.,100.])
        self.assertEqual(result['raw_measurement'],partial)


if __name__ == '__main__':
    unittest.main()
