import json
import pytest
from peleng.colab_support import score_run, export_metres
from peleng.prediction_report import write_prediction_report


def test_scored_and_unscored_reports_share_physical_point_and_missing_policy(tmp_path):
    (tmp_path/'config.json').write_text('{"reference_point":"front"}')
    refs = ['front_face_center','cuboid_center',None,'front_face_center','front_face_center']
    states = ['observed','observed','observed','lost','predicted']
    rows = [dict(source_frame=f, source_time=f/30, world=dict(
        state=states[f], position_cm=[0,0,0], reference_point=refs[f])) for f in range(5)]
    (tmp_path/'observations.jsonl').write_text('\n'.join(map(json.dumps, rows)))
    # Extra final CSV frame must not enter any denominator or chart.
    truth=tmp_path/'truth.csv'
    truth.write_text('Frame;Loc_X;Loc_Y;Loc_Z\n'+''.join(f'{f};0;0;0\n' for f in range(6)))
    scores=score_run(tmp_path,truth,5,'front')
    (tmp_path/'trajectory.csv').write_text(
        'Frame,Time_s,Loc_X,Loc_Y,Loc_Z,state,method,reference_point\n'+
        ''.join(f'{f},{f/30},0,0,0,{states[f]},test,{refs[f] or ""}\n' for f in range(5)))
    export_metres(tmp_path)
    diagnostics=write_prediction_report(tmp_path,5)
    assert scores['matched_frames']==2
    assert scores['coordinate_coverage_pct']==diagnostics['coordinate_coverage_pct']==40
    assert scores['coordinate_frame_mask']==diagnostics['coordinate_frame_mask']==[0,4]
    assert scores['thresholds'][2]['pct_of_valid_reference_frames']==40
    assert scores['longest_missing_frames']==diagnostics['longest_missing_frames']==3
    assert scores['valid_truth_frame_mask']==list(range(5))
    assert scores['scorer_version']==diagnostics['scorer_version']=='named-point-v2'
    assert diagnostics['accuracy'] is None

