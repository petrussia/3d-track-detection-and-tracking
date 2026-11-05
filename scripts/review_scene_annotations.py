"""Exclude known ambiguous negatives and optionally create temporal block splits.

Run a new training from base weights for block splits. Existing adapted weights
have seen some of those frames and cannot be independently validated on them.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

REVIEW_EXCLUSIONS = {(300,3), (450,4), (750,5), (825,5)}


def review(manifest, temporal_blocks=False):
    output = copy.deepcopy(manifest)
    excluded = {tuple(x) for x in output.get('excluded_ambiguous', [])} | REVIEW_EXCLUSIONS
    rows = []
    for row in output['images']:
        if (row['frame'],row['camera_id']) in excluded or row.get('ignore',False):
            continue
        if temporal_blocks:
            frame=row['frame']
            if frame <= 825: row['split']='train'
            elif 1050 <= frame <= 1275: row['split']='val'
            elif frame >= 1500: row['split']='test'
            else:
                excluded.add((frame,row['camera_id']))
                continue
        rows.append(row)
    output['images'] = rows
    output['excluded_ambiguous'] = [list(x) for x in sorted(excluded)]
    output['review'] = dict(date='2026-11-05',
        action='Exclude visible/ambiguous truck fragments from negative examples; no inferred amodal boxes.',
        reviewed_camera_frames=[list(x) for x in sorted(REVIEW_EXCLUSIONS)],
        temporal_blocks=temporal_blocks,
        guard_gap_frames=225 if temporal_blocks else None,
        limitation='Same-scene evaluation, not scene generalization. Block splits require fresh base-weight training.')
    output['split_counts'] = {split:dict(images=len(items), positives=sum(bool(r['boxes']) for r in items),
        moments=len({r['frame'] for r in items})) for split in ('train','val','test')
        for items in [[r for r in rows if r['split']==split]]}
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--temporal-blocks',action='store_true')
    a=p.parse_args()
    result=review(json.loads(a.source.read_text(encoding='utf-8')),a.temporal_blocks)
    result['source_annotations_sha256']=hashlib.sha256(a.source.read_bytes()).hexdigest()
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result['split_counts']))
