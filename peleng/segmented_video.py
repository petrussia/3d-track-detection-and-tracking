"""Decode explicitly numbered video parts in one global frame coordinate system."""
from bisect import bisect_right
import math

import cv2


class SegmentedCapture:
    def __init__(self, paths, factory=None):
        factory = factory or cv2.VideoCapture
        self.parts = [factory(str(path)) for path in paths]
        self.active = 0
        self.offsets, self.counts = [], []
        try:
            metadata = []
            for cap in self.parts:
                if not cap.isOpened():
                    raise ValueError('Cannot open a video segment')
                fps, count, width, height = [cap.get(k) for k in (
                    cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT,
                    cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)]
                if not all(math.isfinite(x) and x > 0 for x in (fps,count,width,height)):
                    raise ValueError('Invalid segment metadata')
                metadata.append((fps,int(count),int(width),int(height)))
                self.offsets.append(sum(self.counts))
                self.counts.append(int(count))
            first=metadata[0]
            if any(abs(m[0]-first[0])>.01 or m[2:]!=first[2:] for m in metadata):
                raise ValueError('Video segments must have equal FPS and dimensions')
            self.metadata={cv2.CAP_PROP_FPS:first[0],cv2.CAP_PROP_FRAME_COUNT:sum(self.counts),
                cv2.CAP_PROP_FRAME_WIDTH:first[2],cv2.CAP_PROP_FRAME_HEIGHT:first[3]}
            self.local_index=-1
        except Exception:
            self.release()
            raise

    def isOpened(self):
        return all(cap.isOpened() for cap in self.parts)

    def get(self, key):
        return self.metadata.get(key, self.parts[self.active].get(key))

    def grab(self):
        if self.local_index+1 >= self.counts[self.active]:
            if self.active+1 >= len(self.parts):
                return False
            self.active += 1
            self.local_index = -1
            if not self.parts[self.active].set(cv2.CAP_PROP_POS_FRAMES, 0):
                return False
        if not self.parts[self.active].grab():
            return False  # Do not conceal an early decode error by skipping a part.
        self.local_index += 1
        return True

    def retrieve(self):
        return self.parts[self.active].retrieve()

    def read(self):
        return self.retrieve() if self.grab() else (False,None)

    def set(self, key, value):
        if key != cv2.CAP_PROP_POS_FRAMES:
            return False
        frame=int(value)
        if frame != value or not 0 <= frame < sum(self.counts):
            return False
        active=bisect_right(self.offsets,frame)-1
        local=frame-self.offsets[active]
        if not self.parts[active].set(key,local):
            return False
        self.active,self.local_index=active,local-1
        return True

    def release(self):
        for cap in self.parts:
            cap.release()
