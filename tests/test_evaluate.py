import json
import csv
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from peleng.evaluate import evaluate_run


def save_fixture(folder, truth_rows, records, fps=None):
    truth = folder/'truth.csv'
    truth.write_text('Frame;Loc_X;Loc_Y;Loc_Z\n'+truth_rows, encoding='cp1251')
    (folder/'observations.jsonl').write_text('\n'.join(json.dumps(x) for x in records), encoding='utf-8')
    if fps is not None:
        (folder/'summary.json').write_text(json.dumps({'source_fps': fps}), encoding='utf-8')
    return truth


def observation(frame, xyz, state='observed', method='triangulation', time=None, **world_fields):
    item = {'source_frame': frame, 'world': {'state': state, 'position_cm': xyz, 'method': method}}
    item['world'].update(world_fields)
    if time is not None:
        item['source_time'] = time
    return item


class EvaluationTests(unittest.TestCase):
    # Catches hiding inaccurate ground assumptions by reporting only later metric fixes.
    def test_method_and_mode_metrics_keep_ground_outputs_in_global_quality_and_coverage(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            truth = save_fixture(folder, '0;0;0;1200\n1;0;0;1200\n2;0;0;1200\n3;0;0;1200\n', [
                observation(0, [300, 400, 0], method='ground_plane', mode='assumed_ground', segment_id=0),
                observation(1, [0, 0, 1200], mode='metric_3d', segment_id=1, mode_reset=True),
                observation(2, [0, 0, 1200], 'coasting', 'prediction', mode='metric_3d', segment_id=1),
                observation(3, None, 'lost', 'none', mode='metric_3d', segment_id=1),
            ])
            result = evaluate_run(folder, truth)
            self.assertEqual(result['matched_observed_steps'], 2)
            self.assertEqual(result['observed_metrics']['three_d']['mean_m'], 6.5)
            self.assertAlmostEqual(result['coverage']['estimate_of_truth'], .75)
            ground = result['metrics_by_mode']['assumed_ground']
            self.assertEqual(ground['observed_metrics']['axes']['z']['mae_m'], 12.)
            self.assertEqual(ground['all_estimate_metrics']['three_d']['rmse_m'], 13.)
            self.assertEqual(ground['coverage']['estimate_of_processed'], .25)
            self.assertEqual(ground['coverage']['estimate_of_truth'], .25)
            self.assertEqual(ground['coverage']['estimate_of_group'], 1.)
            metric = result['metrics_by_mode']['metric_3d']
            self.assertEqual(metric['processed_steps'], 3)
            self.assertEqual(metric['estimate_steps'], 2)
            self.assertEqual(metric['predicted_metrics']['count'], 1)
            self.assertEqual(metric['all_estimate_metrics']['three_d']['rmse_m'], 0.)
            method = result['metrics_by_method']['ground_plane']
            self.assertEqual(method['observed_metrics']['horizontal']['rmse_m'], 5.)

    # Catches either hiding reset jumps globally or pretending a new mode's fix is motion.
    def test_reset_jumps_remain_global_but_are_excluded_from_within_segment_diagnostics(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            truth = save_fixture(folder, '0;0;0;0\n', [
                observation(0, [0, 0, 0], time=0., mode='assumed_ground', segment_id=0),
                observation(1, [10000, 0, 1200], time=1., mode='metric_3d', segment_id=1, mode_reset=True),
                observation(2, [10100, 0, 1200], time=2., mode='metric_3d', segment_id=1),
                observation(3, None, 'lost', time=3., mode='metric_3d', segment_id=2, mode_reset=True),
                observation(4, [50000, 0, 1200], time=4., mode='metric_3d', segment_id=2),
            ])
            result = evaluate_run(folder, truth)
            self.assertEqual(result['discontinuities']['pair_count'], 3)
            self.assertEqual(result['discontinuities']['large_jump_count'], 2)
            self.assertEqual(result['discontinuities']['segment_boundary_pairs'], 2)
            within = result['within_segment_discontinuities']
            self.assertEqual(within['pair_count'], 1)
            self.assertEqual(within['large_jump_count'], 0)
            self.assertEqual(within['max_speed_m_s'], 1.)
            self.assertEqual(within['excluded_boundary_pairs'], 2)
            self.assertEqual(result['segment_reset_steps'], 2)

    # Catches dropping modes in exports or connecting a rendered path across an explicit reset.
    def test_cli_preserves_mode_segments_and_provides_plot_connection_boundaries(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            truth = save_fixture(folder, '0;0;0;0\n1;0;0;0\n2;0;0;0\n3;0;0;0\n', [
                observation(0, [0, 0, 0], mode='assumed_ground', segment_id=0),
                observation(1, [10000, 0, 1200], mode='metric_3d', segment_id=1, mode_reset=True),
                observation(2, [10100, 0, 1200], mode='metric_3d', segment_id=1),
                observation(3, [10200, 0, 1200], mode='metric_3d', segment_id=1, mode_reset=True),
            ])
            process = subprocess.run([sys.executable, '-m', 'peleng.evaluate', '--run', str(folder), '--truth', str(truth)], capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            with (folder/'errors.csv').open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]['mode'], 'assumed_ground')
            self.assertEqual(rows[1]['segment_id'], '1')
            self.assertEqual(rows[1]['mode_reset'], 'True')
            report = (folder/'report.html').read_text(encoding='utf-8')
            embedded = re.search(r'<script type="application/json" id="report-data">(.*?)</script>', report, re.S)
            points = json.loads(embedded.group(1))['points']
            self.assertEqual([p['mode'] for p in points], ['assumed_ground', 'metric_3d', 'metric_3d', 'metric_3d'])
            self.assertEqual([p['segment_id'] for p in points], [0, 1, 1, 1])
            self.assertEqual([p['connect_estimate'] for p in points], [False, False, True, False])

    # Catches ignoring altitude, mixing cm/m, or admitting predictions into legacy errors.
    def test_axis_and_3d_errors_separate_observations_and_predictions(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            truth = save_fixture(folder, '0;0;0;0\n1;0;0;0\n2;0;0;0\n', [
                observation(0, [300, -400, 1200]),
                observation(1, [0, 0, 0], 'coasting', 'prediction'),
                observation(2, [900, 900, 900], 'lost'),
            ], fps=10)
            result = evaluate_run(folder, truth)
            self.assertEqual(result['axis_mae_m'], {'x': 3., 'y': 4., 'z': 12.})
            self.assertEqual(result['axis_rmse_m'], {'x': 3., 'y': 4., 'z': 12.})
            self.assertEqual(result['three_d_rmse_m'], 13.)
            self.assertEqual(result['three_d_p95_m'], 13.)
            self.assertEqual(result['horizontal_mae_m'], 5.)
            self.assertEqual(result['all_estimate_metrics']['three_d']['mean_m'], 6.5)
            self.assertAlmostEqual(result['all_estimate_metrics']['three_d']['rmse_m'], 9.192388155425117)
            self.assertEqual((result['observed_steps'], result['predicted_steps'], result['lost_steps']), (1, 1, 1))
            self.assertEqual(result['estimate_steps'], 2)
            self.assertEqual(result['matched_predicted_steps'], 1)
            self.assertAlmostEqual(result['coverage']['estimate_of_processed'], 2/3)
            self.assertEqual(result['method_counts'], {'triangulation': 1, 'prediction': 1})
            self.assertEqual(result['source_fps'], 10.)

    # Catches accidental nearest-frame/offset alignment and overclaiming truth coverage.
    def test_only_exact_source_frames_are_scored_with_sparse_runs(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            truth = save_fixture(folder, '0;0;0;0\n2;0;0;0\n4;0;0;0\n', [
                observation(1, [100, 0, 0]), observation(2, [300, 400, 0]),
                observation(8, [0, 0, 0]),
            ])
            result = evaluate_run(folder, truth)
            self.assertEqual(result['matched_observed_steps'], 1)
            self.assertEqual(result['horizontal_mae_m'], 5.)
            self.assertEqual(result['coverage']['observed_of_processed'], 1.)
            self.assertAlmostEqual(result['coverage']['observed_of_truth'], 1/3)
            self.assertEqual(result['source_fps'], 30.)

    # Catches invalid frame parsing aborts, NaN poisoning and interpolated/fabricated truth.
    def test_malformed_rows_are_diagnosed_and_not_imputed(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            truth = save_fixture(folder, 'bad;0;0;0\n0;0;0;0\n1;nan;0;0\n2;0\n', [
                observation(0, [300, 400, 1200]), observation(1, [0, 0, 0]),
                observation(2, [0, 0, 0]), observation(3, [float('nan'), 0, 0]),
            ])
            with (folder/'observations.jsonl').open('a') as stream:
                stream.write('\nnot json\n'+json.dumps({'source_frame': 'bad', 'world': {}}))
            result = evaluate_run(folder, truth)
            self.assertEqual(result['truth_frames'], 1)
            self.assertEqual(result['invalid_truth_frames'], [1, 2])
            self.assertEqual(len(result['invalid_truth_rows']), 3)
            self.assertEqual(result['matched_observed_steps'], 1)
            self.assertEqual(result['invalid_estimate_steps'], 1)
            self.assertEqual(len(result['invalid_observation_rows']), 2)
            json.dumps(result, allow_nan=False)

    # Catches crashes/division by zero and reporting zero error when nothing was scored.
    def test_no_overlap_has_null_metrics_and_zero_matched_coverage(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            truth = save_fixture(folder, '10;0;0;0\n', [observation(0, [0, 0, 0])])
            result = evaluate_run(folder, truth)
            self.assertIsNone(result['three_d_rmse_m'])
            self.assertIsNone(result['axis_mae_m']['z'])
            self.assertEqual(result['matched_estimate_steps'], 0)
            self.assertEqual(result['coverage']['estimate_of_truth'], 0.)

    # Catches the old fixed-dt shortcut; 20m over 2s is 10m/s despite sparse frames.
    def test_discontinuities_use_recorded_elapsed_time_and_exclude_lost_positions(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            truth = save_fixture(folder, '0;0;0;0\n', [
                observation(0, [0, 0, 0], time=0.),
                observation(10, [999999, 0, 0], 'lost', time=1.),
                observation(20, [2000, 0, 0], time=2.),
            ], fps=10)
            result = evaluate_run(folder, truth)
            self.assertEqual(result['discontinuities']['pair_count'], 1)
            self.assertEqual(result['discontinuities']['large_jump_count'], 1)
            self.assertEqual(result['discontinuities']['max_speed_m_s'], 10.)
            self.assertEqual(result['discontinuities']['max_jump_m'], 20.)
            self.assertEqual(result['discontinuities']['pairs'][0]['dt_s'], 2.)

    # Catches denominator mismatch: paired quality must use both runs' exact shared frames.
    def test_baseline_comparison_scores_only_paired_frames(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            current, baseline = folder/'current', folder/'baseline'
            current.mkdir()
            baseline.mkdir()
            truth = save_fixture(current, '0;0;0;0\n1;0;0;0\n2;0;0;0\n', [
                observation(0, [300, 400, 0]), observation(1, [0, 0, 0]),
            ])
            save_fixture(baseline, '', [observation(0, [600, 800, 0]), observation(2, [0, 0, 0])])
            result = evaluate_run(current, truth, baseline_run=baseline)
            paired = result['baseline_comparison']['observed_paired']
            self.assertEqual(paired['source_frames'], [0])
            self.assertEqual(paired['current']['horizontal']['mean_m'], 5.)
            self.assertEqual(paired['baseline']['horizontal']['mean_m'], 10.)

    # Catches lost-gap exports, missing CSV axes and unsafe script embedding in a real CLI run.
    def test_cli_writes_portable_report_and_gap_preserving_error_rows(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            truth = save_fixture(folder, '0;0;0;0\n1;0;0;0\n2;0;0;0\n', [
                observation(0, [300, 400, 1200], method='</script><script>alert(1)</script>'),
                observation(1, None, 'lost'),
                observation(2, [0, 0, 0], 'coasting'),
            ])
            result = subprocess.run([sys.executable, '-m', 'peleng.evaluate', '--run', str(folder), '--truth', str(truth)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads((folder/'evaluation.json').read_text())['three_d_rmse_m'], 13.)
            with (folder/'errors.csv').open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]['error_z_m'], '12.0')
            self.assertEqual(rows[0]['error_3d_m'], '13.0')
            self.assertEqual(rows[1]['state'], 'lost')
            self.assertEqual(rows[1]['estimate_x_m'], '')
            self.assertEqual(rows[1]['error_3d_m'], '')
            self.assertEqual(rows[2]['state'], 'coasting')
            html = (folder/'report.html').read_text(encoding='utf-8')
            self.assertIn('<html lang="ru">', html)
            self.assertIn('canvas', html)
            self.assertNotIn('<script src=', html)
            self.assertNotIn('</script><script>alert(1)</script>', html)

    def test_evaluation_skips_corrupt_truth_and_ignores_predicted_positions(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            truth = folder/'truth.csv'
            truth.write_text('Frame;Loc_X;Loc_Y;Loc_Z\n0;0;0;300\n1;авг.13;0;300\n2;0;0;300\n',encoding='cp1251')
            records = [
                {'source_frame':0,'world':{'state':'observed','position_cm':[300,400,0]}},
                {'source_frame':1,'world':{'state':'observed','position_cm':[0,0,0]}},
                {'source_frame':2,'world':{'state':'coasting','position_cm':[9999,9999,0]}},
            ]
            (folder/'observations.jsonl').write_text('\n'.join(json.dumps(x) for x in records))
            result = evaluate_run(folder,truth)
            self.assertEqual(result['matched_observed_steps'],1)
            self.assertEqual(result['invalid_truth_frames'],[1])
            self.assertEqual(result['horizontal_mae_m'],5.)


if __name__ == '__main__':
    unittest.main()
