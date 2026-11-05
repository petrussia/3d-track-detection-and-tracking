"""Compare the same named physical point, with explicit masks and missing policy.

Reads completed journals only. Never modifies historical scores or inputs.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peleng.evaluate import _read_truth, _analyse, _metrics, _errors
from peleng.scoring_contract import coverage_contract, SCORER_VERSION


def compare(runs, expected_frames, reference, truth_path=None):
    truth = _read_truth(truth_path)[0] if truth_path else {}
    truth = {f:p for f,p in truth.items() if 0 <= f < expected_frames}
    result = dict(scorer_version=SCORER_VERSION, reference=reference,
                  expected_frames=expected_frames, ground_truth_available=truth_path is not None,
                  valid_truth_frame_mask=sorted(truth), runs={}, pairwise={})
    records_by_run = {}
    for name, path in runs.items():
        analysis, _, records = _analyse(path, truth, reference, expected_frames)
        if sorted(records) != list(range(expected_frames)):
            raise ValueError(f'{name}: incomplete or nonconsecutive journal')
        available = {f for f,r in records.items() if r['position_cm'] is not None}
        matched = available & truth.keys()
        errors = {f:[(records[f]['position_cm'][axis]-truth[f][axis])/100
                     for axis in range(3)] for f in matched}
        item = coverage_contract(available, expected_frames, reference, analysis['source_fps'])
        item.update(journal_sha256=hashlib.sha256((path/'observations.jsonl').read_bytes()).hexdigest(),
                    matched_frame_mask=sorted(matched), metrics=_metrics(list(errors.values())) if truth_path else None,
                    metrics_by_method=analysis['metrics_by_method'] if truth_path else None,
                    metrics_by_state={s:_metrics(_errors(records,truth,{s})) for s in ['observed','coasting','predicted']} if truth_path else None,
                    discontinuities=analysis['discontinuities'],
                    actual_references=dict(Counter(r['reference_point'] for r in records.values() if r['reference_point'])),
                    thresholds=[dict(tolerance_m=t,success_frames=sum(math.hypot(*e)<=t for e in errors.values()) if truth_path else None,
                        pct_of_valid_truth=100*sum(math.hypot(*e)<=t for e in errors.values())/len(truth) if truth else None)
                        for t in [1,2,5,10,15,20]],
                    episodes={f'{start}-{min(start+99,expected_frames-1)}':
                        dict(available=sum(f in available for f in range(start,min(start+100,expected_frames))),
                             metrics=_metrics([e for f,e in errors.items() if start<=f<start+100]) if truth_path else None)
                        for start in range(0,expected_frames,100)},
                    false_reacquisitions=None,
                    recovery_limitation='Missing-output intervals are not visibility-ground-truth recovery times; false reacquisitions require independent labels.')
        adjacent = [p for p in analysis['discontinuities']['pairs'] if p['to_frame']==p['from_frame']+1]
        item['adjacent_discontinuities'] = dict(pairs_over_10m=[p for p in adjacent if p['jump_m']>10],
            max_step_m=max((p['jump_m'] for p in adjacent), default=None))
        nees = []
        for f in sorted(matched):
            covariance = np.asarray(records[f].get('covariance_cm2'),dtype=float)
            if covariance.shape != (3,3) or not np.isfinite(covariance).all():
                continue
            if not np.allclose(covariance,covariance.T) or np.linalg.eigvalsh(covariance).min()<=0:
                continue
            error_cm = np.asarray(errors[f])*100
            nees.append(float(error_cm @ np.linalg.solve(covariance,error_cm)))
        item['uncertainty_diagnostics'] = dict(covariance_frames=len(nees),
            mean_nees=float(np.mean(nees)) if nees else None,
            fraction_inside_nominal_95_ellipsoid=sum(n<=7.814727903 for n in nees)/len(nees) if nees else None,
            limitation='Diagnostic assumes unbiased Gaussian error for this exact physical point; not a calibrated confidence guarantee. No covariance is inferred from the legacy scalar.')
        result['runs'][name] = item
        records_by_run[name] = records
    names = list(runs)
    for i, first in enumerate(names):
        for second in names[i+1:]:
            a,b = records_by_run[first], records_by_run[second]
            common = {f for f in truth if a[f]['position_cm'] is not None and b[f]['position_cm'] is not None}
            transitions = {}
            for f in sorted(common):
                key = a[f]['method']+' -> '+b[f]['method']
                transitions.setdefault(key, []).append(f)
            result['pairwise'][first+' vs '+second] = dict(common_frame_mask=sorted(common),
                first=_metrics(_errors(a,truth,frames=common)), second=_metrics(_errors(b,truth,frames=common)),
                method_transitions={k:dict(count=len(fs),first=_metrics(_errors(a,truth,frames=set(fs))),
                    second=_metrics(_errors(b,truth,frames=set(fs)))) for k,fs in transitions.items()})
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='append',required=True,help='NAME=RUN_DIRECTORY')
    parser.add_argument('--truth',type=Path)
    parser.add_argument('--reference',choices=['front','center'],default='front')
    parser.add_argument('--expected-frames',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    runs=dict((name,Path(path)) for name,path in (s.split('=',1) for s in args.run))
    result=compare(runs,args.expected_frames,args.reference,args.truth)
    args.output.parent.mkdir(exist_ok=True,parents=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({n:{k:r[k] for k in ['coordinate_coverage_pct','longest_missing_frames','thresholds']} for n,r in result['runs'].items()}))
