"""Clock-controlled engine regression tests; no task footage or model inference."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from peleng.engine import Engine


class Clock:
    def __init__(self):
        self.now = 0.0

    def read(self):
        return self.now


class Capture:
    def __init__(self, _path):
        self.index = -1

    def isOpened(self):
        return True

    def get(self, key):
        return {cv2.CAP_PROP_FPS: 10, cv2.CAP_PROP_FRAME_COUNT: 10,
                cv2.CAP_PROP_FRAME_WIDTH: 64, cv2.CAP_PROP_FRAME_HEIGHT: 36}[key]

    def grab(self):
        self.index += 1
        return self.index < 10

    def retrieve(self):
        return True, np.zeros((36, 64, 3), dtype=np.uint8)

    def release(self):
        pass


class SlowDetector:
    device = "fake"
    name = "synthetic"

    def __init__(self, clock):
        self.clock = clock

    def warmup(self, camera_count):
        pass

    def predict(self, frames):
        self.clock.now += 0.4
        return [[] for _ in frames]


class EngineTests(unittest.TestCase):
    def test_front_ground_bootstrap_never_enters_body_center_observer(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            (folder/'object_transform_CineCameraActor1.csv').write_text(
                'Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw;;;Focal length;10\n'
                '0;0;0;100;0;0;0;;;матрица;40*20мм\n',encoding='cp1251')
            clock=Clock()
            class LargeCapture(Capture):
                def get(self,key):
                    return {cv2.CAP_PROP_FPS:10,cv2.CAP_PROP_FRAME_COUNT:10,
                            cv2.CAP_PROP_FRAME_WIDTH:400,cv2.CAP_PROP_FRAME_HEIGHT:200}[key]
                def retrieve(self):return True,np.zeros((200,400,3),dtype=np.uint8)
            class MovingDetector(SlowDetector):
                def __init__(self,clock):super().__init__(clock);self.frame=-1
                def predict(self,frames):
                    self.frame+=1;self.clock.now+=.4
                    return [[{'bbox':[180+3*self.frame,105,240+3*self.frame,125],
                              'confidence':.8,'label':'truck'}]]
            class Evidence:
                def __init__(self,rig):pass
                def update(self,frames,ids,frame,time,candidates):return candidates
            class NoShape:
                def __init__(self,rig):pass
                def update(self,*args,**kwargs):return None
            engine=Engine(folder,folder/'unused.pt',folder/'outputs',replay_mode='batch',
                          detector_factory=lambda:MovingDetector(clock),recovery_factory=Evidence,shape_factory=NoShape)
            with patch('peleng.engine.time.perf_counter',clock.read), \
                    patch('peleng.engine.cv2.VideoCapture',LargeCapture), \
                    patch('peleng.engine.cv2.imwrite',return_value=True):engine._run()
            state=engine.snapshot()
            self.assertEqual(state['status'],'finished',state['message'])
            records=[json.loads(s) for s in (Path(state['summary']['run_dir'])/'observations.jsonl').read_text().splitlines()]
            last=records[-1]['world']
            self.assertEqual(last['reference_point'],'front_ground_contact')
            self.assertGreater(last['position_cm'][1]-last['body_center_cm'][1],80.)

    def test_rejected_shape_does_not_poison_reference_for_later_image_bearings(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            (folder/'object_transform_CineCameraActor1.csv').write_text(
                'Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw;;;Focal length;35\n'
                '0;0;0;100;0;0;0;;;матрица;23,76*13,365мм\n',encoding='cp1251')
            clock=Clock()
            class Evidence:
                def __init__(self,rig):pass
                def update(self,*args):
                    return [[{'bbox':[22,8,42,28],'confidence':.7,'label':'truck',
                              'source':'foreground','neural_detected':False,'measured':True}]]
            class Shape:
                def __init__(self,rig):pass
                def update(self,frame_index,*args,**kwargs):
                    if frame_index>1:return None
                    return {'position_cm':[1000,0,100],'center_cm':[1000,0,100],
                            'front_cm':[2000 if frame_index==0 else 0,0,100],'front_observable':True,
                            'dimensions_cm':[2000,400,200],'yaw_rad':0. if frame_index==0 else 3.14159,
                            'method':'cuboid_singleview','cameras':[1],
                            'covariance_cm2':np.diag([100.,100.,100.]).tolist(),
                            'metric_depth_observed':False}
            engine=Engine(folder,folder/'unused.pt',folder/'outputs',replay_mode='batch',
                          detector_factory=lambda:SlowDetector(clock),recovery_factory=Evidence,shape_factory=Shape)
            with patch('peleng.engine.time.perf_counter',clock.read), \
                    patch('peleng.engine.cv2.VideoCapture',Capture), \
                    patch('peleng.engine.cv2.imwrite',return_value=True):engine._run()
            state=engine.snapshot()
            self.assertEqual(state['status'],'finished',state['message'])
            records=[json.loads(s) for s in (Path(state['summary']['run_dir'])/'observations.jsonl').read_text().splitlines()]
            self.assertFalse(records[1]['world']['measurement_accepted'])
            self.assertEqual(records[2]['world']['object_center_cm'],[1000.,0.,100.])

    def test_current_image_evidence_reaches_shape_without_counting_as_neural(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder/'object_transform_CineCameraActor1.csv').write_text(
                'Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw;;;Focal length;35\n'
                '0;0;0;100;0;0;0;;;матрица;23,76*13,365мм\n',encoding='cp1251')
            clock = Clock()
            class Evidence:
                def __init__(self, rig): pass
                def update(self, frames, camera_ids, frame_index, timestamp, neural_candidates):
                    return [[{'bbox':[22,8,42,28], 'confidence':.7, 'label':'truck',
                              'source':'foreground', 'neural_detected':False, 'measured':True,
                              'occluded':True,'truncated':True}]]
            class Shape:
                def __init__(self, rig): pass
                def update(self, frame_index, timestamp, observations, **kwargs):
                    if not observations or observations[0].get('source') != 'foreground': return None
                    return {'position_cm':[1000,0,100], 'center_cm':[1000,0,100],
                            'front_cm':[2000,0,100], 'front_observable':True,
                            'dimensions_cm':[2000,400,200], 'yaw_rad':0.,
                            'method':'cuboid_singleview', 'cameras':[1],
                            'covariance_cm2':np.diag([10000.,10000.,10000.]).tolist(),
                            'metric_depth_observed':True}
            engine = Engine(folder, folder/'unused.pt', folder/'outputs', replay_mode='batch',
                            detector_factory=lambda:SlowDetector(clock), recovery_factory=Evidence,
                            shape_factory=Shape, reference='front')
            with patch('peleng.engine.time.perf_counter',clock.read), \
                    patch('peleng.engine.cv2.VideoCapture',Capture), \
                    patch('peleng.engine.cv2.imwrite',return_value=True):
                engine._run()
            state=engine.snapshot()
            self.assertEqual(state['status'],'finished',state['message'])
            summary=state['summary']
            self.assertEqual(summary['steps_with_detection'],0)
            self.assertEqual(summary['steps_with_image_measurement'],10)
            records=[json.loads(s) for s in (Path(summary['run_dir'])/'observations.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['world']['position_cm']==[2000.,0.,100.] for r in records))
            self.assertTrue(all(r['world']['state']=='observed' for r in records))
            self.assertTrue(all(r['world']['reference_point']=='front_face_center' for r in records))
            self.assertTrue(all(r['cameras'][0]['state']=='tracked' for r in records))
            self.assertTrue(all(r['camera_observations'][0].get('occluded') and
                                r['camera_observations'][0].get('truncated') for r in records))

    def test_batch_exports_every_frame_in_order_without_realtime_latency_claim(self):
        import csv
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / 'object_transform_CineCameraActor1.csv').write_text(
                'Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw;;;Focal length;35\n'
                '0;0;0;100;0;0;0;;;матрица;23,76*13,365мм\n', encoding='cp1251')
            clock = Clock()
            engine = Engine(folder, folder/'unused.pt', folder/'outputs',
                            detector_factory=lambda: SlowDetector(clock), replay_mode='batch')
            with patch('peleng.engine.time.perf_counter', clock.read), \
                    patch('peleng.engine.cv2.VideoCapture', Capture), \
                    patch('peleng.engine.cv2.imwrite', return_value=True):
                engine._run()
            state = engine.snapshot()
            self.assertEqual(state['status'], 'finished', state['message'])
            summary = state['summary']
            self.assertEqual(summary['processed_steps'], 10)
            self.assertEqual(summary['skipped_frames_per_camera'], 0)
            self.assertEqual(summary['replay_mode'], 'batch')
            self.assertIsNone(summary['latency_p50_ms'])
            run_dir = Path(summary['run_dir'])
            with (run_dir/'trajectory.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(r['Frame']) for r in rows], list(range(10)))
            self.assertTrue(all(r['state'] == 'lost' and r['Loc_X'] == '' for r in rows))
            self.assertIn('mode', rows[0])
            self.assertIn('reference_point', rows[0])
            self.assertIn('segment_id', rows[0])

    def test_finished_run_counts_tail_frames_arriving_during_final_inference_as_skipped(self):
        # At 10 FPS frames 0/4/8 are processed. Frame 9 arrives during the final
        # inference; it must count as skipped even though no later step runs.
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "object_transform_CineCameraActor1.csv").write_text(
                "Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw;;;Focal length;35\n"
                "0;0;0;100;0;0;0;;;матрица;23,76*13,365мм\n",
                encoding="cp1251",
            )
            clock = Clock()
            engine = Engine(folder, folder / "unused.pt", folder / "outputs",
                            detector_factory=lambda: SlowDetector(clock))
            with patch("peleng.engine.time.perf_counter", clock.read), \
                    patch("peleng.engine.cv2.VideoCapture", Capture), \
                    patch("peleng.engine.cv2.imwrite", return_value=True):
                engine._run()
            state = engine.snapshot()
            self.assertEqual(state["status"], "finished", state["message"])
            summary = state["summary"]
            self.assertEqual(summary["processed_steps"], 3)
            self.assertEqual(summary["available_frames_per_camera"], 10)
            self.assertEqual(summary["skipped_frames_per_camera"], 7)
            self.assertEqual(state["skipped_frames"], 7)

    def test_latency_matches_publication_time_when_preview_storage_is_slow(self):
        # Disk writes take 250 ms. Publishing before that write may legitimately
        # report 400 ms; publishing after it must report at least 650 ms.
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "object_transform_CineCameraActor1.csv").write_text(
                "Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw;;;Focal length;35\n"
                "0;0;0;100;0;0;0;;;матрица;23,76*13,365мм\n",
                encoding="cp1251",
            )
            clock = Clock()
            engine = Engine(folder, folder / "unused.pt", folder / "outputs",
                            detector_factory=lambda: SlowDetector(clock))
            publications = []

            class PublicationState(dict):
                def update(self, *args, **kwargs):
                    super().update(*args, **kwargs)
                    if "source_frame" in kwargs:
                        publications.append((self["source_frame"], clock.now,
                                             self["latency_ms"]))

            engine._state = PublicationState(engine._state)

            def slow_preview_write(_path, _image):
                clock.now += 0.25
                return True

            with patch("peleng.engine.time.perf_counter", clock.read), \
                    patch("peleng.engine.cv2.VideoCapture", Capture), \
                    patch("peleng.engine.cv2.imwrite", slow_preview_write):
                engine._run()
            state = engine.snapshot()
            self.assertEqual(state["status"], "finished", state["message"])
            frame, published_at, latency_ms = publications[0]
            self.assertEqual(frame, 0)
            self.assertAlmostEqual(latency_ms, published_at * 1000)
            records = [json.loads(line) for line in
                       (Path(state["summary"]["run_dir"]) / "observations.jsonl")
                       .read_text().splitlines()]
            self.assertAlmostEqual(records[0]["published_at_s"], published_at)


if __name__ == "__main__":
    unittest.main()
