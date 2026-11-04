"""Pretrained vehicle inference. There is deliberately no training code here."""
import os
from pathlib import Path

# Keep library settings in the project and disable optional online integrations.
os.environ.setdefault('YOLO_CONFIG_DIR',str(Path(__file__).resolve().parent.parent/'outputs/runtime'))
os.environ.setdefault('YOLO_OFFLINE','true')

import numpy as np
import torch
from ultralytics import YOLO, settings


class VehicleDetector:
    def __init__(self, weights: Path, device='auto', image_size=960, confidence=.20):
        settings.update({'sync': False})
        torch.set_num_threads(4)
        self.device = ('cuda:0' if torch.cuda.is_available() else
                       'mps' if torch.backends.mps.is_available() else 'cpu') if device == 'auto' else device
        self.image_size = image_size
        if not 0 < confidence < 1:
            raise ValueError('Confidence must be between zero and one')
        self.confidence = confidence
        if not weights.is_file():
            raise FileNotFoundError(f'Missing model: {weights}. Run ./setup.sh first.')
        self.model = YOLO(str(weights))
        self.name = weights.stem
        self.vehicle_classes = [int(i) for i, name in self.model.names.items()
                                if str(name).lower() in {'car', 'bus', 'truck', 'vehicle', 'грузовик'}]
        if not self.vehicle_classes:
            raise ValueError('Weights must expose a car, bus, truck or vehicle detection class')

    def predict(self, frames):
        results = self.model.predict(frames, imgsz=self.image_size, conf=self.confidence,
                                     classes=self.vehicle_classes, device=self.device,
                                     verbose=False, save=False)
        observations = []
        for result in results:
            boxes = []
            if result.boxes is not None:
                for xyxy, conf, cls in zip(result.boxes.xyxy.cpu().tolist(),
                                           result.boxes.conf.cpu().tolist(),
                                           result.boxes.cls.cpu().tolist()):
                    boxes.append({'bbox': xyxy, 'confidence': conf,
                                  'label': self.model.names[int(cls)]})
            observations.append(boxes)
        return observations

    def warmup(self, camera_count):
        self.predict([np.zeros((1080, 1920, 3), dtype=np.uint8) for _ in range(camera_count)])
        if self.device == 'mps':
            torch.mps.synchronize()
        elif str(self.device).startswith('cuda'):
            torch.cuda.synchronize(self.device)
