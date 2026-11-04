"""A wall-clock replay harness for causal live inference, with a bounded queue of zero.

Recordings stand in for synchronized cameras. At each iteration only the most
recent already-arrived frame is decoded; outdated frames are skipped, not queued.
The target-trajectory CSV is not an input to this module or the detector.
"""
import copy
import csv
import json
import math
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .geometry import CameraRig, estimate_measurement
from .stream import SequentialVideo, due_frame
from .tracking import CameraTrack
from .world import WorldTrack
from .reference import select_reference, reference_from_center, orient_reference
from .initialization import LeadingContact
from .silhouette import center_observations
from .inputs import camera_video


def initial_state():
    return {'status': 'idle', 'source_time': 0., 'source_duration': 0., 'source_fps': 30.,
            'source_frame': -1, 'processed_steps': 0, 'skipped_frames': 0,
            'processing_ms': 0., 'latency_ms': 0., 'effective_fps': 0.,
            'device': '—', 'model': 'YOLO11s', 'message': 'Готов к запуску',
            'cameras': [{'id': i, 'detected': False, 'confidence': None, 'state': 'lost'}
                        for i in range(1, 6)],
            'track': {'state': 'lost', 'position_cm': None, 'method': None, 'trail': []},
            'summary': None}


def annotate(frame, track, camera_id, source_time, projected_point=None):
    image = cv2.resize(frame, (640, 360))
    sx, sy = 640/frame.shape[1], 360/frame.shape[0]
    if track.get('bbox'):
        x1,y1,x2,y2 = track['bbox']
        a,b = (int(x1*sx),int(y1*sy)),(int(x2*sx),int(y2*sy))
        color = (70,220,155) if track['detected'] else (235,190,70) if track.get('measured') else (40,180,245)
        cv2.rectangle(image,a,b,color,2)
        label = f"{track.get('label','vehicle')} {track['confidence']:.2f}"
        if track.get('state') == 'tracked':
            label = 'IMAGE TRACK'
        elif not track['detected']:
            label = f"PREDICTION {track['age_s']:.2f}s"
        cv2.putText(image,label,(a[0],max(42,a[1]-7)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)
    if projected_point is not None:
        point = (round(projected_point[0]*sx), round(projected_point[1]*sy))
        if 0 <= point[0] < 640 and 0 <= point[1] < 360:
            cv2.drawMarker(image,point,(90,100,255),cv2.MARKER_CROSS,18,2)
    cv2.rectangle(image,(0,0),(640,29),(18,25,32),-1)
    cv2.putText(image,f'CAM {camera_id}  |  {source_time:05.2f}s  |  {track["state"].upper()}',
                (10,20),cv2.FONT_HERSHEY_SIMPLEX,.48,(225,235,242),1,cv2.LINE_AA)
    ok,jpeg = cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,80])
    if not ok:
        raise RuntimeError('JPEG encoding failed')
    return jpeg.tobytes(), image


class Engine:
    def __init__(self, data_dir, model_path, output_dir, device='auto', image_size=960,
                 detector_factory=None, replay_mode='live', recovery_factory=None,
                 shape_factory=None, reference='front', confidence=.20):
        self.data_dir = Path(data_dir)
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.device = device
        self.image_size = image_size
        if not 0 < confidence < 1:
            raise ValueError('Confidence must be between zero and one')
        self.confidence = confidence
        self.detector_factory = detector_factory
        if replay_mode not in {'live', 'batch'}:
            raise ValueError('Replay mode must be live or batch')
        self.replay_mode = replay_mode
        if reference not in {'front','center'}:
            raise ValueError('Target reference must be front or center')
        self.recovery_factory = recovery_factory
        self.shape_factory = shape_factory
        self.reference = reference
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._detector = None
        self._state = initial_state()
        self._state['replay_mode']=self.replay_mode
        self._images = {}

    def snapshot(self):
        with self._lock:
            return copy.deepcopy(self._state)

    def image(self, camera_id):
        with self._lock:
            return self._images.get(camera_id)

    def start(self):
        with self._lock:
            if self._thread and self._thread.is_alive():
                return False
            self._state = initial_state()
            self._state.update(status='running', replay_mode=self.replay_mode,
                               message='Загрузка и прогрев модели…')
            self._images = {}
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True, name='peleng-inference')
            self._thread.start()
            return True

    def stop(self):
        self._stop.set()

    def wait(self):
        if self._thread:
            self._thread.join()

    def _publish(self, **changes):
        with self._lock:
            self._state.update(changes)

    def _run(self):
        readers = []
        trace = None
        trajectory = None
        metrics = []
        run_dir = None
        try:
            cv2.setNumThreads(2)
            rig = CameraRig(self.data_dir)
            camera_ids = rig.camera_ids
            metadata = []
            for camera_id in camera_ids:
                path = camera_video(self.data_dir, camera_id)
                capture = cv2.VideoCapture(str(path))
                if not capture.isOpened():
                    raise ValueError(f'Cannot open camera video: {path.name}')
                readers.append(SequentialVideo(capture))
                metadata.append((capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                                 int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            fps = metadata[0][0]
            if fps <= 0 or any(abs(item[0]-fps) > .01 for item in metadata):
                raise ValueError('Camera recordings must have equal valid FPS')
            frame_count = min(item[1] for item in metadata)
            if frame_count <= 0:
                raise ValueError('No video frames')
            if any(m[1:] != metadata[0][1:] for m in metadata):
                raise ValueError('Camera recordings must have equal frame counts and dimensions')
            if min(metadata[0][2:]) <= 0:
                raise ValueError('Invalid video dimensions')
            rig = CameraRig(self.data_dir, width=metadata[0][2], height=metadata[0][3])
            duration = frame_count/fps
            if self._detector is None:
                if self.detector_factory is None:
                    from .detector import VehicleDetector
                    self._detector = VehicleDetector(self.model_path,self.device,self.image_size,
                                                     confidence=self.confidence)
                else:
                    self._detector = self.detector_factory()
            warmup_start = time.perf_counter()
            self._detector.warmup(len(camera_ids))
            warmup_s = time.perf_counter()-warmup_start
            self._publish(source_duration=duration, source_fps=fps, replay_mode=self.replay_mode,
                          device=self._detector.device, model=self._detector.name,
                          message='Обработка поступающих кадров')
            run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
            run_dir = self.output_dir / run_id
            run_dir.mkdir(parents=True)
            trace = (run_dir/'observations.jsonl').open('w',encoding='utf-8')
            trajectory = (run_dir/'trajectory.csv').open('w', encoding='utf-8', newline='')
            trajectory_writer = csv.DictWriter(trajectory, fieldnames=[
                'Frame', 'Time_s', 'Loc_X', 'Loc_Y', 'Loc_Z', 'state', 'method',
                'uncertainty_m', 'metric_depth_observed', 'mode', 'segment_id', 'mode_reset', 'reference_point',
                'Center_X','Center_Y','Center_Z','Front_X','Front_Y','Front_Z'])
            trajectory_writer.writeheader()
            config = {'replay_mode': self.replay_mode, 'model': self._detector.name,
                      'weights': self.model_path.name, 'image_size': self.image_size,
                      'detector_confidence': self.confidence,
                      'source_fps': fps, 'coordinate_units': 'cm',
                      'reference_point': self.reference, 'algorithm': 'causal_image_evidence_and_shared_shape_v2',
                      'world_max_coast_s': .5, 'world_max_bearing_s': 5.,
                      'world_max_speed_cm_s': 5000., 'world_max_correction_floor_cm': 1000.,
                      'fallback_policy': 'current_ground_assumed_then_triangulated_metric_segments',
                      'ground_truth_used_for_inference': False}
            (run_dir/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
            tracks = {i: CameraTrack() for i in camera_ids}
            if self.recovery_factory is None:
                from .recovery import ImageEvidence
                recovery = ImageEvidence(rig)
            else:
                recovery = self.recovery_factory(rig)
            if self.shape_factory is None:
                from .shape import ShapeEstimator
                shape_estimator = ShapeEstimator(rig)
            else:
                shape_estimator = self.shape_factory(rig)
            world_track = WorldTrack()
            body_track = WorldTrack()
            initial_contact = LeadingContact(rig)
            previous_body = None
            previous_timestamp = None
            last_shape = None
            trail = []
            previous_frame = -1
            skipped = 0
            start = time.perf_counter()
            latest_montage = None
            snapshot_at = 0
            while not self._stop.is_set():
                elapsed = time.perf_counter()-start
                if ((self.replay_mode == 'live' and elapsed >= duration)
                        or previous_frame >= frame_count-1):
                    break
                frame_index = (due_frame(elapsed,fps,frame_count) if self.replay_mode == 'live'
                               else previous_frame + 1)
                if frame_index <= previous_frame:
                    self._stop.wait(min(.01,1/fps))
                    continue
                step_start = time.perf_counter()
                decoded = [reader.read_to(frame_index) for reader in readers]
                if any(image is None or index != frame_index for index,image in decoded):
                    raise RuntimeError(f'Decoding failed at source frame {frame_index}')
                frames = [image for _,image in decoded]
                timestamp = frame_index/fps
                candidates = self._detector.predict(frames)
                if len(candidates) != len(camera_ids):
                    raise RuntimeError('Detector returned an unexpected camera count')
                candidates = recovery.update(frames, camera_ids, frame_index, timestamp, candidates)
                states = []
                observations = []
                jpeg_images = {}
                thumbnails = []
                for camera_id,image,boxes in zip(camera_ids,frames,candidates):
                    current = {'id':camera_id,**tracks[camera_id].update(boxes,timestamp)}
                    states.append(current)
                    if current.get('measured', current['detected']):
                        observations.append({'camera_id':camera_id,'bbox':current['bbox'],
                                             'confidence':current['confidence'], 'source':current.get('source','neural'),
                                             'neural_detected':current['detected'], 'measured':True,
                                             'occluded':current.get('occluded',False),
                                             'truncated':current.get('truncated',False),
                                             'quality':current.get('quality',{})})
                dt = timestamp-previous_timestamp if previous_timestamp is not None else 0.
                center_prior = body_track.predict_position(timestamp)
                prior_sigma=(math.sqrt(max(0.,float(np.linalg.eigvalsh(body_track.p[:3,:3]).max())))
                             if body_track.p is not None else 0.)
                center_views,excluded_centers=center_observations(rig,frame_index,observations,
                                                                last_shape,center_prior,prior_sigma)
                raw_center_fix = estimate_measurement(rig, frame_index, center_views,
                                           prior_cm=center_prior,
                                           allow_ground=not body_track.depth_prior_active(timestamp))
                legacy_fix = raw_center_fix
                if last_shape is None and self.reference == 'front':
                    legacy_fix = initial_contact.update(frame_index,timestamp,observations,legacy_fix)
                velocity_hint=None
                if previous_body and previous_body['velocity_cm_s'] is not None and not previous_body['mode_reset']:
                    sigma=math.sqrt(max(0.,float(np.linalg.eigvalsh(body_track.p[3:5,3:5]).max())))
                    if np.linalg.norm(previous_body['velocity_cm_s'][:2]) >= max(100.,3*sigma):
                        velocity_hint=previous_body['velocity_cm_s']
                shape = shape_estimator.update(frame_index, timestamp, observations,
                    prior_cm=center_prior,
                    velocity_cm_s=velocity_hint)
                body_fix=(dict(shape) if shape else reference_from_center(raw_center_fix,last_shape,'center')
                          if last_shape else raw_center_fix)
                if body_fix:
                    body_fix=dict(body_fix)
                    body_fix['reference_point']=('body_contact_proxy' if body_fix['method']=='ground_plane' and not last_shape else 'body_center')
                body=body_track.update(body_fix,timestamp)
                reference_model=last_shape
                if not shape and last_shape and body['state']=='observed' and not body['mode_reset']:
                    reference_model=orient_reference(last_shape,body['velocity_cm_s'],body_track.p[3:5,3:5],dt)
                if shape:
                    fix = select_reference(shape,self.reference)
                else:
                    fix = (reference_from_center(raw_center_fix,reference_model,self.reference)
                           if reference_model is not None else legacy_fix)
                world = world_track.update(fix, timestamp)
                if shape and world['measurement_accepted']:
                    last_shape = dict(fix)
                elif fix and world['measurement_accepted'] and reference_model:
                    last_shape=dict(reference_model)
                    last_shape.update(position_cm=fix['position_cm'],center_cm=fix['center_cm'],
                                      front_cm=fix['front_cm'],reference_point=fix['reference_point'])
                previous_body=body
                previous_timestamp=timestamp
                center = front = None
                if last_shape and world['position_cm'] is not None:
                    shift = np.asarray(world['position_cm'])-np.asarray(last_shape['position_cm'])
                    center = (np.asarray(last_shape['center_cm'])+shift).tolist()
                    front = ((np.asarray(last_shape['front_cm'])+shift).tolist()
                             if last_shape.get('front_cm') is not None else None)
                world.update(object_center_cm=center, front_cm=front,
                             dimensions_cm=last_shape.get('dimensions_cm') if last_shape else None,
                             yaw_rad=last_shape.get('yaw_rad') if last_shape else None,
                             body_center_cm=body['position_cm'],body_velocity_cm_s=body['velocity_cm_s'])
                if world['position_cm'] is not None:
                    if world['mode_reset'] and trail:
                        trail.append(None)
                    trail.append(world['position_cm'][:2])
                    trail = trail[-2000:]
                world['trail'] = trail.copy()
                for camera_id,image,current in zip(camera_ids, frames, states):
                    point = rig.project(camera_id,frame_index,world['position_cm']) if world['position_cm'] is not None else None
                    jpeg,thumbnail = annotate(image,current,camera_id,timestamp,point)
                    jpeg_images[camera_id] = jpeg
                    thumbnails.append(thumbnail)
                # Reserve one tile for metadata, including with 6+ cameras.
                montage_rows = math.ceil((len(thumbnails) + 1) / 3)
                latest_montage = np.zeros((montage_rows*360,1920,3),dtype=np.uint8)
                for i,thumbnail in enumerate(thumbnails):
                    x,y = (i%3)*640,(i//3)*360
                    latest_montage[y:y+360,x:x+640] = thumbnail
                info_x,info_y=(len(thumbnails)%3)*640+20,(len(thumbnails)//3)*360
                cv2.putText(latest_montage,'CAUSAL '+self.replay_mode.upper()+' REPLAY',(info_x,info_y+80),
                            cv2.FONT_HERSHEY_SIMPLEX,.8,(100,220,170),2,cv2.LINE_AA)
                cv2.putText(latest_montage,'No future frames / no trajectory input',(info_x,info_y+120),
                            cv2.FONT_HERSHEY_SIMPLEX,.52,(200,210,220),1,cv2.LINE_AA)
                cv2.putText(latest_montage,'World position: approximate',(info_x,info_y+160),
                            cv2.FONT_HERSHEY_SIMPLEX,.6,(210,215,225),1,cv2.LINE_AA)
                skipped += max(0,frame_index-previous_frame-1)
                with self._lock:
                    finished = time.perf_counter()
                    age_ms = (finished-start-timestamp)*1000 if self.replay_mode == 'live' else None
                    processing_ms = (finished-step_start)*1000
                    metric = {'source_frame':frame_index,'source_time':timestamp,
                              'input_available_s':elapsed if self.replay_mode == 'live' else timestamp,
                              'published_at_s':finished-start if self.replay_mode == 'live' else None,
                              'replay_mode':self.replay_mode,
                              'latency_ms':age_ms,'processing_ms':processing_ms,
                              'excluded_center_observations':excluded_centers,
                              'camera_observations':observations,'cameras':states,'world':world}
                    metrics.append(metric)
                    self._images = jpeg_images
                    self._state.update(source_time=timestamp,source_frame=frame_index,
                                       processed_steps=len(metrics),skipped_frames=skipped,
                                       processing_ms=processing_ms,latency_ms=age_ms,
                                       effective_fps=len(metrics)/(finished-start),
                                       cameras=states,track=world)
                # Saving evidence cannot delay publication of this result. Storage
                # stalls still affect the next frame's cadence/skipped-frame count.
                trace.write(json.dumps(metric,ensure_ascii=False,allow_nan=False)+'\n')
                position = world['position_cm'] or ['', '', '']
                trajectory_writer.writerow({'Frame': frame_index, 'Time_s': timestamp,
                    **dict(zip(('Loc_X','Loc_Y','Loc_Z'), position)), 'state': world['state'],
                    'method': world['method'], 'uncertainty_m': world['uncertainty_m'],
                    'metric_depth_observed': world['metric_depth_observed'],
                    'mode': world['mode'], 'segment_id': world['segment_id'], 'mode_reset': world['mode_reset'],
                    'reference_point': world['reference_point'],
                    **dict(zip(('Center_X','Center_Y','Center_Z'), center or ['', '', ''])),
                    **dict(zip(('Front_X','Front_Y','Front_Z'), front or ['', '', '']))})
                if timestamp >= snapshot_at:
                    cv2.imwrite(str(run_dir/f'preview-{int(snapshot_at):02d}s.jpg'),latest_montage)
                    snapshot_at += 10
                previous_frame = frame_index
            wall_s = time.perf_counter()-start
            status = 'stopped' if self._stop.is_set() else 'finished'
            available_count = (min(frame_count,int(wall_s*fps)+1) if self.replay_mode == 'live'
                               else len(metrics))
            final_skipped = max(0,available_count-len(metrics))
            latencies = [m['latency_ms'] for m in metrics if m['latency_ms'] is not None]
            processing = [m['processing_ms'] for m in metrics]
            summary = {'status':status,'replay_mode':self.replay_mode,'source_fps':fps,
                       'source_frames_per_camera':frame_count,
                       'source_duration_s':duration,'wall_duration_s':wall_s,
                       'warmup_s':warmup_s,'processed_steps':len(metrics),
                       'available_frames_per_camera':available_count,
                       'skipped_frames_per_camera':final_skipped,
                       'output_hz':len(metrics)/wall_s if wall_s else 0.,
                       'latency_p50_ms':float(np.percentile(latencies,50)) if latencies else None,
                       'latency_p95_ms':float(np.percentile(latencies,95)) if latencies else None,
                       'processing_p50_ms':float(np.percentile(processing,50)) if processing else None,
                       'steps_with_detection':sum(any(o.get('neural_detected',True) for o in m['camera_observations']) for m in metrics),
                       'steps_with_image_measurement':sum(bool(m['camera_observations']) for m in metrics),
                       'world_states': {s:sum(m['world']['state']==s for m in metrics)
                                        for s in ('observed','coasting','lost')},
                       'metric_triangulation_steps':sum(m['world']['metric_depth_observed'] and
                            m['world']['method'] in {'triangulated','cuboid_multiview'} for m in metrics),
                       'fresh_metric_depth_steps':sum(m['world']['metric_depth_observed'] for m in metrics),
                       'shape_range_steps':sum(m['world']['measurement_accepted'] and
                            bool((m['world']['raw_measurement'] or {}).get('range_from_shape')) for m in metrics),
                       'rejected_world_updates':sum(m['world']['rejection_reason'] is not None for m in metrics),
                       'detections_by_camera':{str(i):sum(any(o['camera_id']==i and o.get('neural_detected',True) for o in m['camera_observations']) for m in metrics) for i in camera_ids},
                       'device':self._detector.device,'model':self._detector.name,
                       'image_size':self.image_size,'run_dir':str(run_dir),
                       'ground_truth_used_for_inference':False,
                       'coordinate_units_assumed':'cm',
                       'coordinate_reference':self.reference+' of image-derived cuboid; center/front are exported separately; initial ground proxy labeled',
                       'position_limitations':'Rigid cuboid approximates truck shape; dimensions learned only from images. Single-view scale uses past multi-view dimensions. Unknown exact mesh pivot and terrain; uncertainty not independently calibrated.'}
            (run_dir/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
            if latest_montage is not None:
                cv2.imwrite(str(run_dir/'preview.jpg'),latest_montage)
            self._publish(status=status,summary=summary,skipped_frames=final_skipped,
                          message='Прогон завершён' if status=='finished' else 'Остановлено')
        except Exception as exc:
            self._publish(status='error',message=f'{type(exc).__name__}: {exc}')
        finally:
            for reader in readers:
                reader.close()
            if trace:
                trace.close()
            if trajectory:
                trajectory.close()
