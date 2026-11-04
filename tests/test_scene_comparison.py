import json
from scripts.compare_scene_predictions import summarize


def test_reference_changes_and_missing_frames_do_not_become_adjacent_jumps(tmp_path):
    worlds=[
        dict(position_cm=[0,0,0],state='observed',reference_point='cuboid_center'),
        dict(position_cm=[9000,0,0],state='observed',reference_point='ground_contact'),
        dict(position_cm=[10000,0,0],state='observed',reference_point='cuboid_center'),
        dict(position_cm=[12000,0,0],state='observed',reference_point='cuboid_center'),
        dict(position_cm=None,state='lost',reference_point='cuboid_center'),
    ]
    path=tmp_path/'observations.jsonl'
    path.write_text('\n'.join(json.dumps(dict(source_frame=f,world=w)) for f,w in enumerate(worlds)))
    result, frames=summarize(path,5)
    assert frames=={0,2,3}
    assert result['named_coverage_pct']==60
    assert result['max_adjacent_step_m']==20
    assert [s['frame'] for s in result['adjacent_steps_over_10m']]==[3]
    assert [(s['first_frame'],s['last_frame']) for s in result['missing_intervals']]==[(1,1),(4,4)]
