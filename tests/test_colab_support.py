import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from peleng.colab_support import normalize_truth, unpack_bundles, export_metres, score_run

class ColabSupportTests(unittest.TestCase):
    def test_synthetic_metres_convert_once(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);src=d/'truth.csv';dst=d/'normalized.csv'
            src.write_text('Frame,Time_s,X_m,Y_m,Z_m\n0,0,1.5,-2,3\n')
            normalize_truth(src,dst)
            self.assertIn('0;150.0;-200.0;300.0',dst.read_text())

    def test_original_corruption_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);src=d/'truth.csv';dst=d/'normalized.csv'
            raw='Frame;Loc_X;Loc_Y;Loc_Z\n0;авг.13;2;3\n'.encode('cp1251')
            src.write_bytes(raw);normalize_truth(src,dst)
            self.assertEqual(dst.read_bytes(),raw)

    def test_split_archives_verified_and_reused(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);source=d/'in';source.mkdir();dest=d/'out';archive=d/'a.zip'
            with zipfile.ZipFile(archive,'w') as z:z.writestr('datasets/demo/file.txt','example')
            raw=archive.read_bytes();(source/'p1').write_bytes(raw[:20]);(source/'p2').write_bytes(raw[20:])
            (source/'bundle-manifest.json').write_text(json.dumps([{'name':'demo','parts':['p1','p2'],'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}]))
            unpack_bundles(source,dest);unpack_bundles(source,dest)
            self.assertEqual((dest/'datasets/demo/file.txt').read_text(),'example')
            (dest/'.demo.sha256').unlink();(source/'p1').write_bytes(b'bad')
            with self.assertRaises(ValueError):unpack_bundles(source,dest)

    def test_metric_export_preserves_missing_coordinates(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            (d/'trajectory.csv').write_text('Frame,Time_s,Loc_X,Loc_Y,Loc_Z,state,method,reference_point\n0,0,100,-200,300,observed,triangulated,center\n1,0.1,,,,lost,none,center\n')
            export_metres(d)
            with (d/'trajectory_m.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(rows[0]['X_m'],'1.0');self.assertEqual(rows[1]['X_m'],'')

    def test_scores_count_missing_predictions_but_not_extra_truth_row(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);truth=d/'truth.csv'
            truth.write_text('Frame;Loc_X;Loc_Y;Loc_Z\n0;0;0;0\n1;0;0;0\n2;0;0;0\n')
            (d/'config.json').write_text('{"reference_point":"center"}')
            (d/'observations.jsonl').write_text('\n'.join(json.dumps({'source_frame':i,'world':{'state':'observed' if i==0 else 'lost','position_cm':[0,0,0] if i==0 else None}}) for i in [0,1]))
            scores=score_run(d,truth,2,'center')
            self.assertEqual(scores['valid_reference_video_frames'],2)
            self.assertEqual(scores['thresholds'][0]['pct_of_valid_reference_frames'],50)
            self.assertEqual(scores['thresholds'][0]['pct_of_matched_frames'],100)

if __name__=='__main__':unittest.main()
