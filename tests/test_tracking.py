import math
import unittest

from peleng.tracking import CameraTrack


def candidate(bbox=(10, 20, 50, 60), confidence=.8, label='truck'):
    return {'bbox': list(bbox), 'confidence': confidence, 'label': label}


class TrackingTests(unittest.TestCase):
    def test_current_foreground_is_measured_without_being_a_neural_detection(self):
        track = CameraTrack()
        candidate = {'bbox': [10,20,60,70], 'confidence': .7, 'label': 'truck',
                     'source': 'foreground', 'neural_detected': False, 'measured': True}
        current = track.update([candidate], 0)
        self.assertEqual(current['state'], 'tracked')
        self.assertFalse(current['detected'])
        self.assertTrue(current['measured'])
        coast = track.update([], .1)
        self.assertFalse(coast['measured'])
        self.assertEqual(coast['state'], 'coasting')

    def test_crop_proposals_are_not_accepted_as_target_measurements(self):
        hint = {'bbox': [10,20,60,70], 'confidence': .9, 'label': 'truck', 'crop_only': True}
        self.assertEqual(CameraTrack().update([hint], 0)['state'], 'lost')

    def test_short_occlusion_expires_without_future_observations(self):
        track = CameraTrack(max_coast_s=0.5)
        self.assertEqual(track.update([candidate()], 1)['state'], 'detected')
        short = track.update([], 1.2)
        self.assertEqual(short['state'], 'coasting')
        self.assertFalse(short['detected'])
        self.assertEqual(track.update([], 1.6)['state'], 'lost')

    def test_empty_start_is_lost(self):
        self.assertEqual(CameraTrack().update([], 0)['state'], 'lost')

    def test_time_cannot_go_back(self):
        track = CameraTrack()
        track.update([], 1)
        with self.assertRaises(ValueError):
            track.update([], .5)

    def test_nearby_lower_confidence_target_beats_remote_distractor(self):
        track = CameraTrack()
        track.update([candidate()], 0)
        target = candidate((14, 21, 54, 61), .23, 'car')
        distractor = candidate((700, 500, 780, 580), .99)
        result = track.update([distractor, target], .1)
        self.assertEqual(result['bbox'], [14, 21, 54, 61])
        self.assertEqual(result['state'], 'detected')

    def test_remote_candidate_cannot_prevent_loss(self):
        track = CameraTrack()
        track.update([candidate()], 0)
        remote = candidate((700, 500, 780, 580), .99)
        self.assertEqual(track.update([remote], .1)['state'], 'coasting')
        self.assertEqual(track.update([remote], .6)['state'], 'lost')

    def test_similar_center_does_not_allow_abrupt_scale_jump(self):
        track = CameraTrack()
        track.update([candidate()], 0)
        giant = candidate((-50, -40, 110, 120), .99)
        self.assertEqual(track.update([giant], .1)['state'], 'coasting')
        ordinary = candidate((8, 18, 52, 62), .3)
        self.assertEqual(track.update([ordinary], .2)['state'], 'detected')

    def test_initial_acquisition_rejects_tiny_or_weak_boxes(self):
        for box in [candidate((0, 0, 2, 3)), candidate(confidence=.22)]:
            with self.subTest(box=box):
                self.assertEqual(CameraTrack().update([box], 0)['state'], 'lost')

    def test_invalid_boxes_and_confidences_are_ignored(self):
        invalid = [None, {}, candidate((0, 0, 0, 40)),
                   candidate((50, 20, 10, 60)), candidate((0, 0, math.inf, 40)),
                   candidate((0, 0, math.nan, 40)), candidate((0, 0, 40)),
                   candidate(confidence=math.nan), candidate(confidence=math.inf),
                   candidate(confidence=-.1), candidate(confidence=1.1)]
        for box in invalid:
            with self.subTest(box=box):
                self.assertEqual(CameraTrack().update([box], 0)['state'], 'lost')

    def test_consistent_weak_detections_can_initialize_after_confirmation(self):
        track = CameraTrack()
        weak = candidate((10, 20, 50, 60), .228)
        self.assertEqual(track.update([weak], 0)['state'], 'lost')
        remote = candidate((700, 500, 780, 580), .99)
        self.assertEqual(track.update([remote, candidate((12, 20, 52, 60), .24)], .1)['state'],
                         'lost')
        result = track.update([remote, candidate((14, 20, 54, 60), .23)], .2)
        self.assertEqual(result['state'], 'detected')
        self.assertEqual(result['bbox'], [14, 20, 54, 60])

    def test_weak_reacquisition_is_confirmed_and_missing_evidence_resets_it(self):
        track = CameraTrack()
        track.update([candidate()], 0)
        track.update([], .6)
        weak = candidate((700, 500, 740, 540), .228)
        self.assertEqual(track.update([weak], .7)['state'], 'lost')
        self.assertEqual(track.update([], .8)['state'], 'lost')
        self.assertEqual(track.update([weak], .9)['state'], 'lost')
        self.assertEqual(track.update([weak], 1.0)['state'], 'lost')
        self.assertEqual(track.update([weak], 1.1)['state'], 'detected')

    def test_nonfinite_timestamp_does_not_poison_track(self):
        for timestamp in [math.nan, math.inf, -math.inf]:
            with self.subTest(timestamp=timestamp):
                track = CameraTrack()
                track.update([candidate()], 1)
                with self.assertRaises(ValueError):
                    track.update([], timestamp)
                self.assertEqual(track.update([], 1.1)['state'], 'coasting')

    def test_reacquisition_requires_repeated_plausible_detections(self):
        track = CameraTrack()
        track.update([candidate()], 0)
        self.assertEqual(track.update([], .6)['state'], 'lost')
        remote = candidate((700, 500, 740, 540), .9)
        self.assertEqual(track.update([remote], .7)['state'], 'lost')
        recovered = track.update([candidate((702, 501, 742, 541), .8)], .8)
        self.assertEqual(recovered['state'], 'detected')
        self.assertEqual(recovered['bbox'], [702, 501, 742, 541])
        # Velocity from the old target must not teleport this new acquisition.
        self.assertEqual(track.update([], .9)['bbox'], [702, 501, 742, 541])

    def test_reacquisition_confirmation_cannot_use_stale_or_missing_evidence(self):
        track = CameraTrack()
        track.update([candidate()], 0)
        track.update([], .6)
        remote = candidate((700, 500, 740, 540))
        self.assertEqual(track.update([remote], .7)['state'], 'lost')
        self.assertEqual(track.update([], .8)['state'], 'lost')
        self.assertEqual(track.update([remote], .9)['state'], 'lost')
        self.assertEqual(track.update([remote], 1.0)['state'], 'detected')

    def test_motion_prediction_is_used_for_continuation(self):
        track = CameraTrack()
        track.update([candidate()], 0)
        track.update([candidate((30, 20, 70, 60))], .1)
        coast = track.update([], .2)
        self.assertEqual(coast['bbox'], [50, 20, 90, 60])
        result = track.update([candidate((70, 20, 110, 60), .25)], .3)
        self.assertEqual(result['state'], 'detected')
        self.assertEqual(result['bbox'], [70, 20, 110, 60])

    def test_nearly_equal_times_cannot_create_nonfinite_prediction(self):
        track = CameraTrack()
        track.update([candidate()], 0)
        track.update([candidate((30, 20, 70, 60))], 1e-310)
        coast = track.update([], .1)
        self.assertTrue(all(math.isfinite(value) for value in coast['bbox']))
        self.assertLess(coast['bbox'][0], 150)

    def test_reacquisition_cannot_confirm_after_long_gap(self):
        track = CameraTrack()
        track.update([candidate()], 0)
        track.update([], .6)
        remote = candidate((700, 500, 740, 540))
        self.assertEqual(track.update([remote], .7)['state'], 'lost')
        self.assertEqual(track.update([remote], 1.3)['state'], 'lost')
        self.assertEqual(track.update([remote], 1.4)['state'], 'detected')

    def test_configuration_can_admit_smaller_boxes_without_weak_initialization(self):
        track = CameraTrack(min_box_size=2, min_box_area=6,
                            acquisition_confidence=.6, continuation_confidence=.1)
        self.assertEqual(track.update([candidate((0, 0, 3, 3), .7)], 0)['state'],
                         'detected')
        self.assertEqual(track.update([candidate((0, 0, 3, 3), .15)], .1)['state'],
                         'detected')


if __name__ == '__main__':
    unittest.main()
