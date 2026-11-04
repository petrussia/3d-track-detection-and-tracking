"""Sequential decoding: the clock, never model output, controls frame availability."""
import math


def due_frame(elapsed_s: float, fps: float, frame_count: int) -> int:
    if not math.isfinite(elapsed_s) or fps <= 0 or frame_count <= 0:
        raise ValueError('Invalid stream clock')
    return min(frame_count - 1, max(0, math.floor(elapsed_s * fps)))


class SequentialVideo:
    def __init__(self, capture):
        self.capture = capture
        self.index = -1
        self.frame = None

    def read_to(self, target: int):
        if target < self.index:
            raise ValueError('Backward reads are forbidden')
        advanced = False
        while self.index < target:
            if not self.capture.grab():
                return self.index, None
            self.index += 1
            advanced = True
        if advanced:
            ok, frame = self.capture.retrieve()
            self.frame = frame if ok else None
        return self.index, self.frame

    def close(self):
        self.capture.release()
