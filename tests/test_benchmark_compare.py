import json
from pathlib import Path
import tempfile

from scripts.compare_geometry_benchmark import compare, diagnostics


def test_comparison_excludes_changed_reference_and_uses_identical_frames():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        truth = root/'truth.csv'
        truth.write_text('Frame;Loc_X;Loc_Y;Loc_Z\n'+''.join(f'{f};0;0;0\n' for f in range(4)))
        for name, offset, absent in [('way1-baseline',100,None),
                                     ('way1-generic',200,2), ('way1-known',50,3)]:
            folder = root/name/'run'
            folder.mkdir(parents=True)
            rows = []
            for frame in range(4):
                rows.append(dict(source_frame=frame, world=dict(
                    position_cm=None if frame==absent else [offset,0,0],
                    reference_point=('cuboid_center' if name=='way1-baseline' and frame==0 else 'front_face_center'))))
            (folder/'observations.jsonl').write_text('\n'.join(map(json.dumps,rows)))
        result = compare(root,truth,expected_frames=4)
        assert result['common_frames']==1
        for name, error in [('way1-baseline',1.),('way1-generic',2.),('way1-known',.5)]:
            assert result['variants'][name]['common_frame_metrics']['three_d']['rmse_m']==error
            assert result['variants'][name]['named_reference_coverage_pct']==75.
        assert result['variants']['way1-baseline']['excluded_other_references']=={'cuboid_center':1}


def test_diagnostics_separate_prediction_and_do_not_score_missing_truth():
    worlds = {
        0: dict(position_cm=[300,400,0], state='observed', method='cuboid_multiview', uncertainty_m=1),
        1: dict(position_cm=[0,0,1000], state='coasting', method='prediction', uncertainty_m=5),
        2: dict(position_cm=[99999,0,0], state='observed', method='triangulated', uncertainty_m=1),
    }
    result = diagnostics(worlds, {0:[0,0,0], 1:[0,0,0]})
    assert result['by']['state']['observed']['three_d']['rmse_m']==5
    assert result['by']['state']['coasting']['three_d']['rmse_m']==10
    assert [b['metrics']['count'] for b in result['uncertainty_bins']]==[1,0,1,0]
    assert [s['frame'] for s in result['worst_frames']]==[1,0]
    assert result['worst_ten_squared_error_share_pct']==100
