import cv2
import pytest
from peleng.segmented_video import SegmentedCapture
from peleng.stream import SequentialVideo
from peleng.inputs import camera_video_segments


class Capture:
    def __init__(self,name):
        self.base=int(name); self.index=-1
    def isOpened(self): return True
    def get(self,key):
        return {cv2.CAP_PROP_FPS:30.,cv2.CAP_PROP_FRAME_COUNT:3,
                cv2.CAP_PROP_FRAME_WIDTH:100,cv2.CAP_PROP_FRAME_HEIGHT:80}.get(key,0)
    def set(self,key,value): self.index=int(value)-1; return True
    def grab(self): self.index+=1; return self.index<3
    def retrieve(self): return True,self.base+self.index
    def release(self): pass


def test_global_frames_cross_parts_and_seek_without_duplicates():
    cap=SegmentedCapture(['0','3','6'],Capture)
    assert cap.offsets==[0,3,6]
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT)==9
    reader=SequentialVideo(cap)
    assert [reader.read_to(i)[1] for i in range(9)]==list(range(9))
    assert reader.read_to(9)[1] is None
    assert cap.set(cv2.CAP_PROP_POS_FRAMES,3)
    assert cap.read()==(True,3)
    assert cap.set(cv2.CAP_PROP_POS_FRAMES,7)
    assert cap.read()==(True,7)


def test_numbered_parts_reject_missing_parts_and_ambiguous_whole(tmp_path):
    one=tmp_path/'Way2.FinalImage.CineCameraActor5_part1.mp4'; one.touch()
    three=tmp_path/'Way2.FinalImage.CineCameraActor5_part3.mp4'; three.touch()
    with pytest.raises(ValueError,match='consecutively'):
        camera_video_segments(tmp_path,5)
    two=tmp_path/'Way2.FinalImage.CineCameraActor5_part2.mp4'; two.touch()
    assert camera_video_segments(tmp_path,5)==[one,two,three]
    (tmp_path/'Way2.FinalImage.CineCameraActor5.mp4').touch()
    with pytest.raises(ValueError,match='Ambiguous'):
        camera_video_segments(tmp_path,5)
