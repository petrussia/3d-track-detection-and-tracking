from pathlib import Path
import pytest
from peleng.geometry import CameraRig
from peleng.inputs import camera_video
from peleng.prediction_report import write_prediction_report


def test_scene2_calibration_and_video(tmp_path):
    camera = tmp_path / 'object_transform_CineCameraActor1.csv'
    camera.write_text('Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw;Focal_Length_mm\n0;0;0;0;0;0;0;35\n1;0;0;0;0;0;0;35\n')
    (tmp_path / 'matrix_size.txt').write_text('23,76 mm * 13,365 mm')
    rig = CameraRig(tmp_path)
    assert rig.project(1, 0, (100, 10, 0))[0] == pytest.approx(1242.8282828282827)
    video = tmp_path / 'Way2.FinalImage.CineCameraActor1.mp4'
    video.touch()
    assert camera_video(tmp_path, 1) == video
    (tmp_path / 'Way1.FinalImage.CineCameraActor1.mp4').touch()
    with pytest.raises(ValueError, match='Ambiguous'):
        camera_video(tmp_path, 1)
    camera.write_text(camera.read_text().replace('1;0;0;0;0;0;0;35', '1;0;0;0;0;0;0;50'))
    with pytest.raises(ValueError, match='Variable focal'):
        CameraRig(tmp_path)


def test_no_truth_report_preserves_missing_points(tmp_path):
    (tmp_path / 'trajectory_m.csv').write_text('Frame,Time_s,X_m,Y_m,Z_m,state,method,reference_point\n0,0,1,2,3,measured,multi,center\n1,0.033,,,,lost,none,center\n2,0.066,2,3,4,predicted,filter,center\n')
    result = write_prediction_report(tmp_path, 3)
    assert result['coordinate_frames'] == 2
    assert result['coordinate_coverage_pct'] == pytest.approx(200/3)
    assert result['accuracy'] is None
    assert (tmp_path / 'report.html').read_text(encoding='utf-8').count('<polyline') == 6
